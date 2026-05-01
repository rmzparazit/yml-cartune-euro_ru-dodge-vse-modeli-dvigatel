import os
import re
import sys
import time
import asyncio
import hashlib
import json
import argparse
import aiohttp
import shutil
import subprocess
import backoff
import tempfile
from io import BytesIO
from PIL import Image
from datetime import datetime
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, RateLimitError, APIError
from typing import Optional, Any
from lxml import etree
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

try:
    import requests
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==========================================
# НАСТРОЙКИ ФАЙЛОВ
# ==========================================
CONFIG_FILE = "feed_settings.json"
CACHE_FILE = "feed_cache.json"

# ==========================================
# 1. СХЕМЫ ДАННЫХ И МОДЕЛИ
# ==========================================


class RawExtractedProduct(BaseModel):
    h1_title: str = Field(default="Без названия")
    brand: str = Field(default="Unknown")
    price_raw: Any = Field(default=0, alias="price")
    oldprice_raw: Any = Field(default=0, alias="oldprice")
    currency: str = Field(default="RUB")
    images: list[str] = Field(default_factory=list)
    specs: dict[str, Any] = Field(default_factory=dict)
    available: bool = Field(default=True)
    category_name: str = Field(default="Каталог")
    category_usp: str = Field(default="")
    description_usp: str = Field(default="")
    sales_notes: str = Field(default="")
    custom_labels: list[str] = Field(default_factory=list)


class TransformedProduct(BaseModel):
    offer_id: str
    url: str
    name: str
    type_prefix: str
    vendor: str
    model_name: str
    price: str
    oldprice: str
    currency: str
    images: list[str]
    description: str
    sales_notes: str
    specs: dict[str, str]
    custom_labels: list[str]
    available: str
    category_id: str


class CategoryCollection(BaseModel):
    category_id: str
    name: str
    url: str
    picture: str
    description: str

# ==========================================
# УТИЛИТА: ВАЛИДАЦИЯ ИЗОБРАЖЕНИЙ
# ==========================================


async def validate_image_url(url: str, session: aiohttp.ClientSession) -> bool:
    try:
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.read()
                img = Image.open(BytesIO(data))
                width, height = img.size
                return width >= 450 and height >= 450
    except Exception:
        return False
    return False

# ==========================================
# 2. КЭШ-МЕНЕДЖЕР
# ==========================================


class CacheManager:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=4, ensure_ascii=False)

    def get_md5(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def check_cache(self, url: str, markdown_content: str) -> tuple[bool, Optional[dict]]:
        current_md5 = self.get_md5(markdown_content)
        if url in self.cache and self.cache[url].get("md5") == current_md5:
            return True, self.cache[url].get("raw_data")
        return False, None

    def update_cache(self, url: str, markdown_content: str, raw_data: dict):
        self.cache[url] = {
            "md5": self.get_md5(markdown_content),
            "raw_data": raw_data,
            "last_seen": datetime.now().isoformat()
        }
        self.save()

    def get_all_cached_urls(self):
        return set(self.cache.keys())

    def get_raw_data(self, url: str):
        return self.cache.get(url, {}).get("raw_data")

# ==========================================
# 3. ФАЗА РАЗВЕДКИ (DISCOVERY)
# ==========================================


class DiscoveryAgent:
    @staticmethod
    def analyze_and_group_links(base_url: str, html_content: str) -> dict:
        groups = {}
        base_parsed = urlparse(base_url)
        soup = BeautifulSoup(html_content, 'lxml')

        stop_words = ['login', 'cart', 'korzina', 'tel:', 'mailto:', '.jpg', '.png', 'policy',
                      'consent', 'contacts', 'kontakt', 'pro-o-nas', 'about', 'oplata',
                      'dostavka', 'rezerv', 'vozvrat', 'otzyvy', 'faq', 'help', 'garantiya',
                      '+7', '8800', 'javascript:', 'whatsapp', 'viber', 'tg://', 'auth', 'register']

        for a in soup.find_all('a', href=True):
            href = a.get('href')
            if not href:
                continue
            parsed_href = urlparse(href)

            if parsed_href.netloc and parsed_href.netloc != base_parsed.netloc:
                continue

            href_path = parsed_href.path.lower()
            if any(word in href_path for word in stop_words):
                continue

            clean_href = href.split('#')[0].split('?')[0]
            if clean_href in ['/', '', base_parsed.netloc]:
                continue

            full_url = urljoin(base_url, clean_href)
            path_parts = [p for p in href_path.split('/') if p]

            if not path_parts:
                signature = "/"
            elif len(path_parts) == 1:
                signature = f"/{path_parts[0]}/*"
            else:
                signature = f"/{'/'.join(path_parts[:-1])}/*"

            title = ""
            parent_card = a.find_parent(['div', 'li', 'article'], class_=lambda c: c and any(
                x in c.lower() for x in ['product', 'item', 'good', 'card']))
            if parent_card:
                texts = [text for text in parent_card.stripped_strings if text]
                if texts:
                    title = " | ".join(texts[:3])

            if not title:
                title = a.get_text(strip=True) or a.get('title', '')
                if not title and a.find('img'):
                    title = a.find('img').get('alt', '') or a.find(
                        'img').get('title', '')

            title = ' '.join(title.split()) if title else "Без названия"

            if signature not in groups:
                groups[signature] = {}
            if full_url not in groups[signature] or len(title) > len(groups[signature].get(full_url, "")):
                groups[signature][full_url] = title

        result = {}
        for sig, links in groups.items():
            if len(links) > 0:
                result[sig] = [{"url": k, "title": v}
                               for k, v in links.items()]

        def sort_key(item):
            sig, links = item
            is_product = 1 if any(
                x in sig for x in ['product', 'item', 'detail', 'catalog/']) else 0
            return (is_product, len(links))

        return dict(sorted(result.items(), key=sort_key, reverse=True))

# ==========================================
# 4. ФАЗА ИЗВЛЕЧЕНИЯ (КЛАССИКА И ИИ)
# ==========================================


class ClassicScraper:
    @staticmethod
    def extract_product_data(url: str, html_content: str, markdown_content: str = "") -> RawExtractedProduct:
        soup = BeautifulSoup(html_content, 'lxml')

        h1_el = soup.find(['h1', 'h2'], class_=re.compile(
            r'title|name|head', re.I)) or soup.find('h1')
        h1_title = h1_el.get_text(strip=True) if h1_el else "Без названия"

        price_raw = "0"
        currency = "RUB"

        currency_map = {
            '₽': 'RUB', 'руб': 'RUB', 'rub': 'RUB', 'р.': 'RUB', 'р': 'RUB',
            '$': 'USD', 'usd': 'USD', '€': 'EUR', 'eur': 'EUR',
            '₸': 'KZT', 'kzt': 'KZT', 'byn': 'BYN', 'бел': 'BYN'
        }

        curr_pattern = r'(₽|руб|rub|р\.|р|\$|usd|€|eur|₸|kzt|byn)'
        price_regex = r'(?<!\d)(\d[\d\s.,\xa0]{0,15})\s*' + curr_pattern

        def find_price() -> tuple[str, str]:
            if markdown_content:
                clean_md = markdown_content.replace(
                    '&nbsp;', '').replace('&#160;', '')
                matches_md = re.findall(
                    price_regex, clean_md.lower(), re.IGNORECASE)
                for val_str, curr_str in matches_md:
                    clean_val = re.sub(r'[^\d]', '', val_str)
                    if clean_val and int(clean_val) > 0:
                        return clean_val, currency_map.get(curr_str.strip().lower(), 'RUB')

            for node in soup.find_all(class_=re.compile(r'price|cost|amount', re.I)):
                text = node.get_text(separator=' ').lower()
                matches = re.findall(price_regex, text, re.IGNORECASE)
                for val_str, curr_str in matches:
                    clean_val = re.sub(r'[^\d]', '', val_str)
                    if clean_val and int(clean_val) > 0:
                        return clean_val, currency_map.get(curr_str.strip().lower(), 'RUB')

            script_texts = [s.get_text()
                            for s in soup.find_all('script') if s.get_text()]
            for script_text in script_texts:
                match = re.search(
                    r'[\'"]?price[\'"]?\s*:\s*(\d+(?:\.\d+)?)', script_text, re.IGNORECASE)
                if match:
                    val = re.sub(r'[^\d]', '', match.group(1))
                    if val and int(val) > 0:
                        return val, 'RUB'

            texts = [t for t in soup.stripped_strings if t.parent.name not in [
                'script', 'style', 'head', 'noscript']]
            visible_text = ' '.join(texts).replace(
                '\xa0', '').replace('&nbsp;', '').lower()
            matches = re.findall(price_regex, visible_text, re.IGNORECASE)
            for val_str, curr_str in matches:
                clean_val = re.sub(r'[^\d]', '', val_str)
                if clean_val and int(clean_val) > 0:
                    return clean_val, currency_map.get(curr_str.strip().lower(), 'RUB')

            return "0", "RUB"

        price_raw, currency = find_price()

        oldprice_raw = "0"
        for oldprice_node in soup.find_all(class_=re.compile(r'old-price|old_price|price-old|crossed', re.I)):
            clean_str = re.sub(r'[^\d]', '', oldprice_node.get_text())
            if clean_str and int(clean_str) > 0:
                oldprice_raw = clean_str
                break

        images = []
        for img_container in soup.find_all(['div', 'a', 'img', 'picture'], class_=re.compile(r'img|image|slider|gallery|photo', re.I)):
            img = img_container if img_container.name == 'img' else img_container.find(
                'img')
            if img:
                src = img.get('src') or img.get(
                    'data-src') or img.get('data-lazy')
                if src and src.startswith('http'):
                    images.append(src)

        specs = {}
        brand = "Unknown"

        for row in soup.find_all('tr'):
            cols = row.find_all(['td', 'th'])
            if len(cols) == 2:
                k, v = cols[0].get_text(
                    strip=True), cols[1].get_text(strip=True)
                specs[k] = v

        for item in soup.find_all(['li', 'div'], class_=re.compile(r'param|property|feature|attribute', re.I)):
            name_el = item.find(['div', 'span'], class_=re.compile(
                r'name|title|label', re.I))
            val_el = item.find(
                ['div', 'span'], class_=re.compile(r'value|val', re.I))

            if name_el and val_el:
                specs[name_el.get_text(strip=True)] = val_el.get_text(
                    strip=True)
            else:
                text = item.get_text(strip=True)
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts[0]) < 30:
                        specs[parts[0].strip()] = parts[1].strip()

        for k, v in specs.items():
            if k.lower() in ['марка', 'производитель', 'бренд']:
                brand = v

        desc_usp = ""
        desc_container = soup.find(['div', 'section'], class_=re.compile(
            r'comment|description|detail-text|about', re.I))
        if desc_container:
            desc_usp = desc_container.get_text(separator=' ', strip=True)

        sales_notes = ""
        delivery_nodes = soup.find_all(string=re.compile(
            r'(предоплат|наложенн|оплат|картой|наличн|доставк|тк|отправк)', re.I))
        for node in delivery_nodes:
            parent = getattr(node, 'parent', None)
            if parent and parent.name not in ['script', 'style']:
                text = parent.get_text(strip=True)
                if 5 < len(text) <= 50 and (any(char.isdigit() for char in text) or "%" in text or "карт" in text.lower() or "налич" in text.lower() or "тк " in text.lower()):
                    sales_notes = text
                    break

        cat_name = "Каталог"
        breadcrumbs = soup.find_all(['span', 'li', 'a', 'div'], class_=re.compile(
            r'breadcrumb|bx-breadcrumb|nav', re.I))
        if len(breadcrumbs) > 1:
            cat_name = breadcrumbs[-1].get_text(strip=True)

        custom_labels = []
        if brand != "Unknown":
            custom_labels.append(brand)
        if cat_name != "Каталог":
            custom_labels.append(cat_name)

        page_text_lower = html_content.lower()
        if "б/у" in page_text_lower or "бывш" in page_text_lower or "пробег" in page_text_lower:
            custom_labels.append("Б/У")
        if "контрактн" in page_text_lower:
            custom_labels.append("Контрактный")

        return RawExtractedProduct(
            h1_title=h1_title, brand=brand, price_raw=price_raw, oldprice_raw=oldprice_raw,
            currency=currency, images=list(set(images)), specs=specs, category_name=cat_name,
            category_usp=cat_name, description_usp=desc_usp, sales_notes=sales_notes, custom_labels=custom_labels[
                :5]
        )


class AIScraper:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider.lower()
        if self.provider == "deepseek":
            self.llm_client = AsyncOpenAI(
                api_key=api_key, base_url="https://llms.dotpoin.com/v1/")
        elif self.provider == "openai":
            self.llm_client = AsyncOpenAI(api_key=api_key)
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Обновите пакет: pip install google-genai")
            self.gemini_client = genai.Client(api_key=api_key)

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def extract_product_data(self, url: str, markdown_content: str) -> RawExtractedProduct:
        prompt = f"""
        Ты Senior Data Extractor. URL: {url}
        Верни JSON:
        - h1_title: Главный заголовок
        - brand: Производитель
        - price: Цена (только цифры)
        - currency: Валюта (RUB, USD, EUR, KZT, BYN).
        - oldprice: Старая цена (если есть).
        - images: URL изображений.
        - specs: Объект характеристик. Исключи 'Производитель'.
        - available: true/false.
        - category_name: Имя категории.
        - category_usp: Продающее УТП категории (до 80 симв). БЕЗ точки в конце.
        - description_usp: Техническое описание. БЕЗ эмодзи. УТП до 80 симв.
        - sales_notes: условия оплаты/доставки до 50 симв. Если нет четких сроков, цен или методов оплаты - верни пустую строку "".
        - custom_labels: массив из 1-5 строк (Бренд, Категория, Б/У если применимо).
        Markdown: {markdown_content[:35000]}
        """

        if self.provider in ["openai", "deepseek"]:
            model_name = "deepseek-chat" if self.provider == "deepseek" else "gpt-4o-mini"
            response = await self.llm_client.chat.completions.create(
                model=model_name, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "You output strict JSON exactly matching the schema."},
                          {"role": "user", "content": prompt}], temperature=0.1
            )
            raw_json = json.loads(response.choices[0].message.content)
        elif self.provider == "gemini":
            response = await self.gemini_client.aio.models.generate_content(
                model='gemini-2.5-flash', contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.1),
            )
            raw_json = json.loads(response.text)

        return RawExtractedProduct(**raw_json)

# ==========================================
# 5. ФАЗА ТРАНСФОРМАЦИИ
# ==========================================


class DataTransformer:
    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def smart_truncate(text: str, max_length: int) -> str:
        text = str(text).strip()
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]
        # Вырезаем висящие предлоги и слова в конце
        text = re.sub(r'\s+(и|в|на|с|от|до|за|по|к|из|у|без|возможна|продажа)$',
                      '', text, flags=re.IGNORECASE)
        # ЖЕСТКАЯ очистка: выжигаем любые знаки препинания в конце всей строки
        text = re.sub(r'[.,:;!?\-\s]+$', '', text)
        return text.strip()

    @staticmethod
    def generate_numeric_id(text: str) -> str:
        return str(int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16))[:90]

    @staticmethod
    def clean_emojis_and_specials(text: str) -> str:
        clean_text = re.sub(r'[^\w\s.,!?\-:;/"\'()]', '', str(text))
        return re.sub(r'\s+', ' ', clean_text).strip()

    @staticmethod
    def compress_commercial_text(text: str, max_length: int = 175) -> str:
        if not text:
            return ""

        text = re.sub(r'(?i)id\s*товара\s*\d+.*$', '', text)

        fluff_patterns = [
            r'(?i)комментарий от продавца[:\-\s]*',
            r'(?i)внимание[:\-\s]*',
            r'(?i)уважаемые покупатели[:\-\s]*',
            r'(?i)мы предоставляем полный пакет документов.*?учёт[:\-\s]*',
            r'(?i)копия грузовой.*?деклараци[ии].*?(?=\.|\n|$)',
            r'(?i)договор купли-продажи.*?(?=\.|\n|$)',
            r'(?i)есть аукционный лист.*?(?=\.|\n|$)',
            r'(?i)предоставим подробное фото.*?видео.*?(?=\.|\n|$)',
            r'(?i)возможна проверка эндоскопом.*?(?=\.|\n|$)',
            r'(?i)цена указана за.*?фото.*?(?=\.|\n|$)',
            r'(?i)описание товара[:\-\s]*',
            r'(?i)возможна продажа без навесного.*?([.\n]|$)',
            r'(?i)возможна продажа.*?([.\n]|$)',
            r'(?i)(Номер по производителю|Производитель|Марка|Модель|Год|Кузов|Артикул)[\s:]*$'
        ]
        for pattern in fluff_patterns:
            text = re.sub(pattern, '', text)

        abbreviations = {
            r'(?i)\bbmw\b': 'BMW',
            r'(?i)\baudi\b': 'AUDI',
            r'(?i)\bоригинальный\b': 'ориг.',
            r'(?i)\bоригинальные\b': 'ориг.',
            r'(?i)\bоригинал\b': 'ориг.',
            r'(?i)\bпробег\b': 'проб.',
            r'(?i)\bкилометров\b': 'км',
            r'(?i)\bдвигатель\b': 'ДВС',
            r'(?i)\bавтомобиль\b': 'авто',
            r'(?i)\bавтомобиля\b': 'авто',
            r'(?i)\bнавесного оборудования\b': 'навесного',
            r'(?i)\bработоспособность\b': 'работу',
            r'(?i)\bсостояние\b': 'сост.',
            r'(?i)\bгарантия\b': 'гарант.',
            r'(?i)\bв наличии\b': 'в нал.',
            r'(?i)\bбез дефектов\b': 'без деф.',
            r'(?i)\bгрузы\b': 'товары'
        }
        for k, v in abbreviations.items():
            text = re.sub(k, v, text)

        # Улучшенная чистка пунктуации
        # Убираем "Дефект: -" -> "Дефект: "
        text = re.sub(r':\s*-\s*', ': ', text)
        text = re.sub(r'\(\s+', '(', text)         # Убираем пробел после (
        text = re.sub(r'\s+\)', ')', text)         # Убираем пробел перед )
        text = re.sub(r'\.{2,}', '.', text)        # Убираем многоточия
        text = re.sub(r'\.\s*-\s*\.', '.', text)   # Убираем ". -."
        text = re.sub(r'\s*-\s*\.', '.', text)     # Убираем " -."
        text = re.sub(r'\.\s*-\s*', '. ', text)    # Убираем ". -"
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Прижимаем знаки к словам
        text = re.sub(r'\s+', ' ', text).strip()   # Убираем двойные пробелы
        text = re.sub(r'^[.,\s\-]+', '', text)     # Убираем знаки в начале

        return DataTransformer.smart_truncate(text, max_length)

    @staticmethod
    def parse_universal_price(raw_price_val: Any) -> float:
        if not raw_price_val:
            return 0.0
        if isinstance(raw_price_val, (int, float)):
            return float(raw_price_val)
        clean_str = re.sub(r'[^\d.,]', '', str(raw_price_val).replace(
            '\xa0', '').replace(' ', '')).replace(',', '.')
        parts = clean_str.split('.')
        if len(parts) > 2:
            clean_str = ''.join(parts[:-1]) + '.' + parts[-1]
        try:
            return float(clean_str) if clean_str else 0.0
        except ValueError:
            return 0.0

    def apply_title_prefix(self, title: str) -> str:
        prefix = self.config.get("title_prefix", "").strip()
        if not prefix:
            return title
        words = title.split()
        if not words:
            return title
        if not words[0].isupper():
            words[0] = words[0].lower()
        return f"{prefix} {' '.join(words)}"

    def transform(self, raw: RawExtractedProduct, url: str, category_id_map: dict) -> TransformedProduct:
        valid_price = self.parse_universal_price(raw.price_raw)
        valid_oldprice = self.parse_universal_price(raw.oldprice_raw)

        if self.config.get("auto_oldprice", True) and valid_oldprice == 0 and valid_price > 0:
            valid_oldprice = valid_price * 1.10

        clean_h1 = self.clean_emojis_and_specials(raw.h1_title)
        prefixed_name = self.apply_title_prefix(clean_h1)
        final_name = self.smart_truncate(prefixed_name, 56)

        words = clean_h1.split()
        type_prefix = words[0] if words else "Товар"

        vendor_clean = self.clean_emojis_and_specials(raw.brand)[:50]
        if vendor_clean.lower() == "unknown" or not vendor_clean:
            vendor_clean = "Noname"

        model_str = clean_h1
        model_str = re.sub(rf'^{re.escape(type_prefix)}\s*',
                           '', model_str, flags=re.IGNORECASE)
        model_str = re.sub(rf'{re.escape(vendor_clean)}\s*',
                           '', model_str, flags=re.IGNORECASE).strip()
        if not model_str:
            model_str = raw.specs.get('Модель', 'Без модели')

        final_desc = self.compress_commercial_text(
            self.clean_emojis_and_specials(raw.description_usp), 175)

        default_offer_desc = self.config.get(
            "default_offer_description", "").strip()
        if not final_desc:
            final_desc = self.smart_truncate(
                default_offer_desc, 175) if default_offer_desc else final_name

        default_sales = self.config.get("default_sales_notes", "").strip()
        extracted_sales = self.clean_emojis_and_specials(
            raw.sales_notes).strip()

        final_sales = ""
        if extracted_sales:
            final_sales = self.smart_truncate(extracted_sales, 50)
        elif default_sales:
            final_sales = self.smart_truncate(default_sales, 50)

        safe_specs = {}
        stop_param_keys = ['производитель', 'бренд', 'марка', 'модель']
        stop_param_values = ['none', 'null', 'n/a',
                             'не указан', 'нет', '-', '', 'стандартный']

        for k, v in raw.specs.items():
            clean_k = self.clean_emojis_and_specials(str(k)).strip()
            clean_v = self.clean_emojis_and_specials(str(v)).strip()
            if clean_k.lower() not in stop_param_keys and clean_v.lower() not in stop_param_values:
                safe_specs[clean_k] = clean_v

        cat_name_clean = self.clean_emojis_and_specials(raw.category_name)[:56]
        cat_id = category_id_map.get(
            cat_name_clean, self.generate_numeric_id(cat_name_clean)[:10])

        price_str = "0" if valid_price == 0 else f"{valid_price:.2f}"
        oldprice_str = "0" if valid_oldprice == 0 else f"{valid_oldprice:.2f}"

        return TransformedProduct(
            offer_id=self.generate_numeric_id(url),
            url=url,
            name=final_name,
            type_prefix=type_prefix,
            vendor=vendor_clean,
            model_name=model_str,
            price=price_str,
            oldprice=oldprice_str if valid_oldprice > valid_price else "",
            currency=raw.currency,
            images=raw.images[:5],
            description=final_desc,
            sales_notes=final_sales,
            specs=safe_specs,
            custom_labels=[self.clean_emojis_and_specials(
                lbl)[:175] for lbl in raw.custom_labels[:5]],
            available="true" if raw.available else "false",
            category_id=cat_id
        )

# ==========================================
# 6. ФАЗА СЕРИАЛИЗАЦИИ YML
# ==========================================


class YMLBuilder:
    def __init__(self, config: dict, date_str: str):
        self.shop_name = config.get("shop_name", "Shop")
        self.company_name = config.get("company_name", "Company")
        self.site_url = config.get("site_url", "https://example.com")
        self.feed_mode = config.get("feed_mode", "1")
        self.duplicate_offers = config.get("duplicate_offers", False)
        self.config = config
        self.date_str = date_str

    def _wrap_text(self, el, text, is_description=False):
        if not text:
            return

        # Удаляем непечатаемые символы (0-31), разрешая только Tab(9), LF(10), CR(13)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Автоматическая обертка CDATA только для description и только если есть спецсимволы
        if is_description and self.config.get("auto_cdata", True) and re.search(r'[&<>\'"]', text):
            el.text = etree.CDATA(text)
        else:
            # Ручная замена кавычек, остальное ( < > & ) lxml экранирует сам
            text = text.replace('"', '&quot;').replace("'", '&apos;')
            el.text = text

    def build_empty_feed(self, output_path: str):
        root = etree.Element('yml_catalog', date=self.date_str)
        shop = etree.SubElement(root, 'shop')
        etree.SubElement(shop, 'name').text = self.shop_name
        etree.SubElement(shop, 'company').text = self.company_name
        etree.SubElement(shop, 'url').text = self.site_url
        currencies = etree.SubElement(shop, 'currencies')
        etree.SubElement(currencies, 'currency', id="RUB", rate="1")
        etree.SubElement(shop, 'categories')
        etree.SubElement(shop, 'offers')
        tree = etree.ElementTree(root)
        tree.write(output_path, pretty_print=True,
                   xml_declaration=True, encoding='utf-8')

    def build_feed(self, products: list[TransformedProduct], collections: list[CategoryCollection], output_path: str):
        if not products and not collections:
            self.build_empty_feed(output_path)
            return

        root = etree.Element('yml_catalog', date=self.date_str)
        shop = etree.SubElement(root, 'shop')

        self._wrap_text(etree.SubElement(shop, 'name'), self.shop_name)
        self._wrap_text(etree.SubElement(shop, 'company'), self.company_name)
        etree.SubElement(shop, 'url').text = self.site_url

        used_currencies = set(p.currency for p in products if p.currency)
        if "RUB" not in used_currencies:
            used_currencies.add("RUB")

        currencies_el = etree.SubElement(shop, 'currencies')
        for c_code in sorted(list(used_currencies)):
            rate = "1" if c_code == "RUB" else "CBRF"
            etree.SubElement(currencies_el, 'currency', id=c_code, rate=rate)

        categories_el = etree.SubElement(shop, 'categories')
        for coll in collections:
            cat = etree.SubElement(
                categories_el, 'category', id=coll.category_id)
            self._wrap_text(cat, coll.name)

        if self.feed_mode in ['1', '2'] and products:
            offers_el = etree.SubElement(shop, 'offers')
            for prod in products:
                offer = etree.SubElement(
                    offers_el, 'offer', id=prod.offer_id, available=prod.available, type="vendor.model")

                # ТЕГ NAME ТЕПЕРЬ СТРОГО ПЕРВЫЙ
                self._wrap_text(etree.SubElement(offer, 'name'), prod.name)
                self._wrap_text(etree.SubElement(offer, 'url'), prod.url)

                etree.SubElement(offer, 'price').text = prod.price
                if prod.oldprice and prod.oldprice != "0":
                    etree.SubElement(offer, 'oldprice').text = prod.oldprice

                etree.SubElement(offer, 'currencyId').text = prod.currency
                etree.SubElement(offer, 'categoryId').text = prod.category_id
                etree.SubElement(offer, 'collectionId').text = prod.category_id

                for img_url in prod.images:
                    self._wrap_text(etree.SubElement(
                        offer, 'picture'), img_url)

                self._wrap_text(etree.SubElement(
                    offer, 'typePrefix'), prod.type_prefix)
                self._wrap_text(etree.SubElement(offer, 'vendor'), prod.vendor)
                self._wrap_text(etree.SubElement(
                    offer, 'model'), prod.model_name)

                # Description с умной оберткой CDATA
                self._wrap_text(etree.SubElement(
                    offer, 'description'), prod.description, is_description=True)

                if prod.sales_notes:
                    self._wrap_text(etree.SubElement(
                        offer, 'sales_notes'), prod.sales_notes)

                for i, lbl in enumerate(prod.custom_labels):
                    self._wrap_text(etree.SubElement(
                        offer, f'custom_label_{i}'), lbl)

                for key, val in prod.specs.items():
                    if val:
                        param_el = etree.SubElement(offer, 'param', name=key)
                        self._wrap_text(param_el, val)

        if self.feed_mode in ['1', '3']:
            collections_el = etree.SubElement(shop, 'collections')
            def_coll_desc = self.config.get(
                "default_collection_description", "").strip()

            for coll in collections:
                collection = etree.SubElement(
                    collections_el, 'collection', id=coll.category_id)
                self._wrap_text(etree.SubElement(collection, 'url'), coll.url)
                self._wrap_text(etree.SubElement(
                    collection, 'name'), coll.name)

                c_desc = coll.description if coll.description else def_coll_desc
                if not c_desc:
                    c_desc = coll.name

                self._wrap_text(etree.SubElement(collection, 'description'),
                                DataTransformer.smart_truncate(c_desc, 81), is_description=True)
                if coll.picture:
                    self._wrap_text(etree.SubElement(
                        collection, 'picture'), coll.picture)

            if (self.duplicate_offers and self.feed_mode == '1') or self.feed_mode == '3':
                for prod in products:
                    collection = etree.SubElement(
                        collections_el, 'collection', id=f"col_{prod.offer_id}")
                    self._wrap_text(etree.SubElement(
                        collection, 'url'), prod.url)
                    self._wrap_text(etree.SubElement(
                        collection, 'name'), f"{prod.type_prefix} {prod.vendor} {prod.model_name}".strip())

                    c_desc = prod.description if prod.description else def_coll_desc
                    if not c_desc:
                        c_desc = prod.name

                    self._wrap_text(etree.SubElement(collection, 'description'),
                                    DataTransformer.smart_truncate(c_desc, 81), is_description=True)
                    if prod.images:
                        self._wrap_text(etree.SubElement(
                            collection, 'picture'), prod.images[0])

        tree = etree.ElementTree(root)
        tree.write(output_path, pretty_print=True,
                   xml_declaration=True, encoding='utf-8')

# ==========================================
# ИНТЕРАКТИВНЫЙ РЕДАКТОР ФИДОВ (YML Editor)
# ==========================================


class YMLEditor:
    @staticmethod
    def edit_feed_interactive(config):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*50)
        print("📝 РЕДАКТОР СГЕНЕРИРОВАННЫХ ФИДОВ")
        print("="*50)

        xml_files = [f for f in os.listdir('.') if f.endswith('.xml')]
        if not xml_files:
            print("❌ XML фиды не найдены в текущей директории.")
            input("Нажмите Enter...")
            return

        for i, f in enumerate(xml_files, 1):
            print(f"[{i}] {f}")

        try:
            choice = int(input("\nВыберите номер файла: ").strip())
            target_file = xml_files[choice-1]
        except (ValueError, IndexError):
            return

        try:
            parser = etree.XMLParser(strip_cdata=False)
            tree = etree.parse(target_file, parser)
            root = tree.getroot()
        except Exception as e:
            print(f"Ошибка чтения файла: {e}")
            input("Нажмите Enter...")
            return

        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Файл: {target_file}")
            print("1. Редактировать элементы вручную")
            print("2. Пересобрать текущий фид с сайта (Парсинг)")
            print("0. Сохранить и выйти")

            action = input("\nВыбор: ").strip()

            if action == '0':
                tree.write(target_file, pretty_print=True,
                           xml_declaration=True, encoding='utf-8')
                print("💾 Изменения сохранены!")
                time.sleep(1)
                break

            elif action == '2':
                print(
                    "\n⚠️ ВНИМАНИЕ: Все ваши ручные изменения в этом файле будут перезаписаны!")
                confirm = input("Продолжить? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', 'д', 'да']:
                    urls_to_reparse = set()
                    for offer in root.findall('.//offer'):
                        url_tag = offer.find('url')
                        if url_tag is not None and url_tag.text:
                            urls_to_reparse.add(url_tag.text)

                    if urls_to_reparse:
                        # Direct URL Mode - Исключает парсинг посторонних ссылок
                        asyncio.run(run_parser(config, auto_mode=False, override_urls=list(
                            urls_to_reparse), override_filename=target_file, direct_urls_mode=True))

                        gh = input(
                            "\nЖелаете отправить обновленный фид в GitHub? (y/n, Enter=Да): ").strip().lower()
                        if gh in ['y', 'yes', 'д', 'да', '']:
                            # Находим правильную мета-инфу (URL) для генерации правильного имени папки
                            t_url = config.get(
                                "target_urls", ["https://example.com"])[0]
                            export_dir = build_github_snapshot(
                                config, target_file, target_url=t_url)
                            auto_create_and_push_github(
                                config, export_dir, target_url=t_url)
                        return
                    else:
                        print(
                            "❌ В файле не найдено валидных ссылок на товары для повторного парсинга.")
                        time.sleep(2)
                continue

            elif action == '1':
                mode = input(
                    "\nЧто редактируем? (1 - Офферы, 2 - Коллекции, 0 - Отмена): ").strip()
                if mode not in ['1', '2']:
                    continue

                elements = root.findall(
                    './/offer') if mode == '1' else root.findall('.//collection')
                if not elements:
                    print("Элементы не найдены.")
                    time.sleep(1)
                    continue

                while True:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"--- СПИСОК ЭЛЕМЕНТОВ ---")
                    for i, el in enumerate(elements, 1):
                        name = el.find('name')
                        name_text = name.text if name is not None else "Без имени"
                        print(f"[{i}] {name_text[:60]}")

                    try:
                        el_choice = int(input(
                            "\nВыберите номер элемента для редактирования (или 0 для возврата): ").strip())
                        if el_choice == 0:
                            break
                        target_el = elements[el_choice-1]
                    except (ValueError, IndexError):
                        continue

                    while True:
                        os.system('cls' if os.name == 'nt' else 'clear')
                        n_tag = target_el.find('name')
                        print(
                            f"--- Редактирование: {n_tag.text if n_tag is not None else 'Без имени'} ---")

                        tags = ['name', 'price', 'description', 'sales_notes'] if mode == '1' else [
                            'name', 'description']
                        for idx, t_name in enumerate(tags, 1):
                            t_node = target_el.find(t_name)
                            t_val = t_node.text if t_node is not None else "[ПУСТО]"
                            print(f"[{idx}] {t_name.upper()}: {t_val[:50]}...")

                        print("\n[0] Назад")

                        try:
                            tag_choice = int(
                                input("Выберите поле для изменения: ").strip())
                            if tag_choice == 0:
                                break
                            t_name = tags[tag_choice-1]
                        except (ValueError, IndexError):
                            continue

                        t_node = target_el.find(t_name)
                        current_val = t_node.text if t_node is not None else ""
                        print(f"\nТекущее значение: {current_val}")
                        new_val = input(
                            "Новое значение (Enter - оставить): ").strip()

                        if new_val:
                            if t_name == 'description':
                                limit = 175 if mode == '1' else 81
                                new_val = DataTransformer.smart_truncate(
                                    new_val, limit)
                            if t_name == 'sales_notes':
                                new_val = DataTransformer.smart_truncate(
                                    new_val, 50)

                            if t_node is None:
                                t_node = etree.SubElement(target_el, t_name)

                            is_cdata = '<![CDATA[' in etree.tostring(
                                t_node, encoding='unicode')
                            t_node.text = etree.CDATA(
                                new_val) if is_cdata else new_val
                            print("✅ Изменено!")
                            time.sleep(1)

# ==========================================
# GITHUB ИНТЕГРАЦИЯ
# ==========================================


def get_target_repo_name(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '').replace('.', '_')
    path_parts = [p for p in parsed.path.split('/') if p]
    path_str = "-".join(path_parts) if path_parts else "main"
    repo_name = f"yml-{domain}-{path_str}"
    return re.sub(r'[^a-zA-Z0-9_\-]', '-', repo_name)[:100]


def get_static_filename(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '').replace('.', '_')
    path_parts = [p for p in parsed.path.split('/') if p]
    path_str = "_".join(path_parts) if path_parts else "main"
    return f"{domain}_{path_str}_feed.xml"


def generate_feed_filename(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '').replace('.', '_')
    path_parts = [p for p in parsed.path.split('/') if p]
    path_str = "_".join(path_parts) if path_parts else "main"
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{domain}_{path_str}_{date_str}_feed.xml"


def build_github_snapshot(config, output_filename, export_dir_name=None, target_url=None):
    static_filename = get_static_filename(
        target_url) if target_url else "feed.xml"

    if target_url:
        parsed = urlparse(target_url)
        domain = parsed.netloc.replace('www.', '').replace('.', '_')
        path_parts = [p for p in parsed.path.split('/') if p]
        path_str = "_".join(path_parts) if path_parts else "main"
        export_dir = export_dir_name or f"export_{domain}_{path_str}"
    else:
        domain = urlparse(config["site_url"]).netloc.split(
            ':')[0].replace('www.', '').replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        export_dir = export_dir_name or f"export_{domain}_{timestamp}"

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(os.path.join(export_dir, ".github",
                "workflows"), exist_ok=True)

    frozen_config = config.copy()
    frozen_config["openai_api_key"] = ""
    frozen_config["gemini_api_key"] = ""
    frozen_config["deepseek_api_key"] = ""
    frozen_config["github_pat"] = ""
    frozen_config["output_file"] = static_filename
    save_config(frozen_config, os.path.join(export_dir, "feed_settings.json"))

    shutil.copy(__file__, os.path.join(export_dir, "feed.py"))

    if os.path.exists(output_filename):
        shutil.copy(output_filename, os.path.join(export_dir, static_filename))
    else:
        open(os.path.join(export_dir, static_filename), 'a').close()

    open(os.path.join(export_dir, "feed_cache.json"), 'a').write("{}")

    cron_schedule = config.get("cron_schedule", "0 3 * * *")

    workflow = f"""name: Auto-Update YML Feed
on:
  schedule:
    - cron: '{cron_schedule}' 
  workflow_dispatch: 

jobs:
  build-and-update:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip' 
          
      - name: Cache Playwright browsers
        uses: actions/cache@v4
        with:
          path: ~/.cache/ms-playwright
          key: playwright-${{{{ runner.os }}}}-${{{{ hashFiles('**/feed.py') }}}}
          restore-keys: |
            playwright-${{{{ runner.os }}}}-
            
      - name: Install dependencies
        run: |
          pip install crawl4ai pydantic openai lxml beautifulsoup4 aiohttp playwright Pillow backoff requests
          playwright install chromium

      - name: Run Scraper
        env:
          DEEPSEEK_API_KEY: ${{{{ secrets.DEEPSEEK_API_KEY }}}}
        run: python feed.py --auto
        
      - name: Commit and Push
        run: |
          git config --global user.name 'github-actions[bot]'
          git config --global user.email 'github-actions[bot]@users.noreply.github.com'
          git add {static_filename} feed_cache.json feed_settings.json
          git commit -m "Auto-update YML feed [skip ci]" || echo "No changes"
          git push
"""
    with open(os.path.join(export_dir, ".github", "workflows", "update_feed.yml"), "w", encoding="utf-8") as f:
        f.write(workflow)

    print(f"\n✅ Снапшот для GitHub собран: {export_dir}")
    return export_dir


def edit_github_feed_settings(main_config, target_dir):
    config_path = os.path.join(target_dir, "feed_settings.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            feed_config = json.load(f)
    except Exception:
        print("❌ Не удалось прочитать настройки фида.")
        return

    while True:
        clear_screen()
        print(f"=== РЕДАКТИРОВАНИЕ НАСТРОЕК: {target_dir} ===")
        print(
            f"1. Расписание (CRON): {feed_config.get('cron_schedule', '0 3 * * *')}")
        print(
            f"2. УТП (sales_notes): {feed_config.get('default_sales_notes', 'Авто-поиск')}")
        print(
            f"3. Описание (description): {feed_config.get('default_offer_description', 'Авто-поиск')}")
        print("4. 💾 Сохранить настройки и отправить в GitHub")
        print("0. Назад")

        choice = input("\nВыбор: ").strip()
        if choice == '1':
            print(
                "\nПримеры: '0 3 * * *' (каждый день в 3:00), '0 */12 * * *' (каждые 12 часов)")
            cron = input("Введите CRON (Enter оставить текущий): ").strip()
            if cron:
                feed_config['cron_schedule'] = cron
        elif choice == '2':
            sales = input("\nВведите sales_notes: ").strip()
            feed_config['default_sales_notes'] = sales
        elif choice == '3':
            desc = input("\nВведите description: ").strip()
            feed_config['default_offer_description'] = desc
        elif choice == '4':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(feed_config, f, indent=4, ensure_ascii=False)

            workflow_path = os.path.join(
                target_dir, ".github", "workflows", "update_feed.yml")
            if os.path.exists(workflow_path):
                with open(workflow_path, 'r', encoding='utf-8') as wf:
                    w_content = wf.read()
                w_content = re.sub(
                    r"cron:\s*'.*?'", f"cron: '{feed_config.get('cron_schedule', '0 3 * * *')}'", w_content)
                with open(workflow_path, 'w', encoding='utf-8') as wf:
                    wf.write(w_content)

            print("\n⬆️ Отправка изменений в GitHub...")
            try:
                subprocess.run(
                    ["git", "-C", target_dir, "add", "."], check=True)
                status = subprocess.run(
                    ["git", "-C", target_dir, "status", "--porcelain"], capture_output=True, text=True).stdout
                if status.strip():
                    subprocess.run(
                        ["git", "-C", target_dir, "commit", "-m", "Update feed settings"], check=True)
                    res = subprocess.run(["git", "-C", target_dir, "push"])
                    if res.returncode == 0:
                        print("✅ Настройки успешно обновлены на GitHub!")
                    else:
                        print("❌ Ошибка при отправке push.")
                else:
                    print("ℹ️ Изменений не найдено.")
            except Exception as e:
                print(f"❌ Ошибка Git: {e}")
            time.sleep(2)
            break
        elif choice == '0':
            break


def auto_create_and_push_github(config, export_dir, target_url=None):
    token = config.get("github_pat", "")
    if not token:
        print("❌ Ошибка: Не задан GitHub PAT. Настройте его в меню (Пункт 12).")
        return

    t_url = target_url if target_url else config.get(
        "target_urls", ["https://example.com"])[0]
    repo_name = get_target_repo_name(t_url)
    static_filename = get_static_filename(t_url)

    try:
        import requests
    except ImportError:
        print("❌ Ошибка: Для работы GitHub API необходима библиотека requests (pip install requests)")
        return

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    print(f"🔍 Проверка репозитория {repo_name} на GitHub...")
    user_resp = requests.get("https://api.github.com/user", headers=headers)
    if user_resp.status_code != 200:
        print("❌ Ошибка авторизации. Проверьте ваш PAT токен.")
        return

    username = user_resp.json()["login"]
    repo_url = f"https://{token}@github.com/{username}/{repo_name}.git"
    raw_url = f"https://raw.githubusercontent.com/{username}/{repo_name}/main/{static_filename}"

    create_resp = requests.post(
        "https://api.github.com/user/repos",
        headers=headers,
        json={"name": repo_name, "private": False,
              "description": "Automated YML Feed"}
    )

    if create_resp.status_code == 201:
        print(f"✅ Репозиторий {repo_name} успешно создан!")
    elif create_resp.status_code == 422:
        print(f"ℹ️ Репозиторий {repo_name} уже существует. Используем его.")
    else:
        print(f"❌ Ошибка создания репозитория: {create_resp.text}")
        return

    print("⬆️ Отправка файлов в GitHub...")

    temp_base = tempfile.mkdtemp()
    temp_repo_path = os.path.join(temp_base, os.path.basename(export_dir))

    try:
        shutil.copytree(export_dir, temp_repo_path)

        git_dir = os.path.join(temp_repo_path, '.git')
        if os.path.exists(git_dir):
            shutil.rmtree(git_dir, ignore_errors=True)

        subprocess.run(["git", "-C", temp_repo_path, "init"], check=True)
        subprocess.run(["git", "-C", temp_repo_path, "config",
                       "user.email", "bot@yml.local"], check=True)
        subprocess.run(["git", "-C", temp_repo_path, "config",
                       "user.name", "YML Bot"], check=True)
        subprocess.run(["git", "-C", temp_repo_path, "add", "."], check=True)

        status = subprocess.run(["git", "-C", temp_repo_path, "status",
                                "--porcelain"], capture_output=True, text=True).stdout
        if status.strip():
            subprocess.run(["git", "-C", temp_repo_path, "commit",
                           "-m", "Auto-setup YML pipeline"], check=True)

        subprocess.run(["git", "-C", temp_repo_path,
                       "branch", "-M", "main"], check=True)
        subprocess.run(["git", "-C", temp_repo_path,
                       "remote", "add", "origin", repo_url])
        subprocess.run(["git", "-C", temp_repo_path, "remote",
                       "set-url", "origin", repo_url])

        res = subprocess.run(
            ["git", "-C", temp_repo_path, "push", "-u", "origin", "main", "--force"])
        if res.returncode == 0:
            print(
                f"✅ Файлы успешно отправлены! Ссылка на репозиторий: https://github.com/{username}/{repo_name}")
            print(
                f"\n🔗 ПРЯМАЯ ССЫЛКА НА XML ДЛЯ ЯНДЕКС.ДИРЕКТ:\n👉 {raw_url}\n")

            config["github_repo_url"] = f"https://github.com/{username}/{repo_name}"
            config["github_raw_url"] = raw_url
            save_config(config)

            export_config_path = os.path.join(export_dir, "feed_settings.json")
            if os.path.exists(export_config_path):
                try:
                    with open(export_config_path, 'r', encoding='utf-8') as f:
                        exp_conf = json.load(f)
                    exp_conf["github_repo_url"] = f"https://github.com/{username}/{repo_name}"
                    exp_conf["github_raw_url"] = raw_url
                    with open(export_config_path, 'w', encoding='utf-8') as f:
                        json.dump(exp_conf, f, indent=4, ensure_ascii=False)
                except:
                    pass
        else:
            print("❌ Ошибка при выполнении git push.")
    except Exception as e:
        print(f"❌ Ошибка Git: {e}")
    finally:
        shutil.rmtree(temp_base, ignore_errors=True)


def push_export_to_github(config, export_dir_name=None, target_url=None):
    if export_dir_name:
        latest_dir = export_dir_name
    else:
        export_dirs = [d for d in os.listdir(
            '.') if os.path.isdir(d) and d.startswith('export_')]
        if not export_dirs:
            print("❌ Сначала сгенерируйте экспорт в меню.")
            time.sleep(2)
            return
        latest_dir = sorted(export_dirs)[-1]

    auto_create_and_push_github(config, latest_dir, target_url)

# ==========================================
# ОРКЕСТРАТОР ПАЙПЛАЙНА (CLI)
# ==========================================


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "use_ai_extraction": False,
        "active_ai_provider": "deepseek",
        "deepseek_api_key": "",
        "target_urls": ["https://cartune-euro.ru/bmw/vse-modeli/dvigatel/"],
        "shop_name": "My AI Store",
        "company_name": "AI E-commerce LLC",
        "site_url": "https://mysite.com",
        "auto_cdata": True,
        "skip_empty_price": True,
        "auto_oldprice": True,
        "split_feeds": False,
        "interactive_selection": True,
        "auto_github_export": False,
        "github_pat": "",
        "github_raw_url": "",
        "cron_schedule": "0 3 * * *",
        "default_sales_notes": "",
        "default_offer_description": "",
        "default_collection_description": "",
        "feed_mode": "1",
        "duplicate_offers": False,
        "title_prefix": "",
        "auto_product_signatures": [],
        "auto_category_signatures": []
    }


def save_config(config, filename=CONFIG_FILE):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


async def run_parser(config, auto_mode=False, override_urls=None, override_filename=None, direct_urls_mode=False):
    use_ai = config.get("use_ai_extraction", False)

    if use_ai:
        api_key = config.get("deepseek_api_key", "")
        if not api_key:
            print(f"❌ Ошибка: Не найден API-ключ DeepSeek.")
            if not auto_mode:
                input("\nНажмите Enter...")
            return
        scraper = AIScraper(provider="deepseek", api_key=api_key)
        mode_text = "AI (DEEPSEEK)"
    else:
        scraper = ClassicScraper()
        mode_text = "CLASSIC (Эвристика + Защита от сбоев сети)"

    t_urls = override_urls if override_urls else config.get("target_urls", [])
    if not t_urls:
        return

    print(f"\n🚀 Запуск Агента. Режим: {mode_text}...")

    transformer = DataTransformer(config)
    discovery_agent = DiscoveryAgent()
    cache_manager = CacheManager()

    skip_empty_price = config.get("skip_empty_price", True)
    split_feeds = config.get("split_feeds", False)
    interactive_selection = config.get("interactive_selection", True)

    target_groups = []
    if direct_urls_mode:
        target_groups = [t_urls]
    else:
        if split_feeds and not auto_mode and not override_urls:
            for u in t_urls:
                target_groups.append([u])
        else:
            target_groups.append(t_urls)

    for urls in target_groups:
        transformed_products = []
        collections_map = {}
        category_id_map = {}

        url_queue = list(urls)
        visited_urls = set()
        crawled_product_urls = set()

        semaphore = asyncio.Semaphore(1)

        if override_filename:
            output_filename = override_filename
        elif auto_mode:
            output_filename = config.get("output_file", "feed.xml")
        else:
            output_filename = generate_feed_filename(urls[0])

        if interactive_selection and not override_urls and not auto_mode and not direct_urls_mode:
            chosen_prod_sigs = set()
            chosen_cat_sigs = set()
        else:
            chosen_prod_sigs = set(config.get("auto_product_signatures", []))
            chosen_cat_sigs = set(config.get("auto_category_signatures", []))

        async def safe_fetch(crawler_instance, target_url, delay=1.5, retries=3):
            for attempt in range(retries):
                try:
                    result = await crawler_instance.arun(
                        url=target_url,
                        bypass_cache=True,
                        magic=True,
                        delay_before_return_html=delay
                    )
                    if result and result.html:
                        return result
                except Exception as e:
                    error_msg = str(e)
                    if "ERR_NAME_NOT_RESOLVED" in error_msg or "Proxy direct failed" in error_msg or "Timeout" in error_msg:
                        if not auto_mode:
                            print(
                                f"  [Сеть] Споткнулся (Сбой соединения). Ждем 3 сек. Попытка {attempt + 1}/{retries}...")
                        await asyncio.sleep(3.0)
                    else:
                        if not auto_mode:
                            print(f"  [Ошибка Crawl4AI] {e}")
                        break
            return None

        async def process_single_product(product_url: str, parent_category_url: str, crawler_instance):
            async with semaphore:
                try:
                    crawled_product_urls.add(product_url)
                    is_cached, cached_raw_data = cache_manager.check_cache(
                        product_url, "")

                    await asyncio.sleep(2.5)

                    prod_result = await safe_fetch(crawler_instance, product_url, delay=2.0)
                    if not prod_result or not prod_result.html:
                        if not auto_mode:
                            print(
                                f"  ❌ Пропуск: Не удалось загрузить страницу {product_url}")
                        return

                    is_cached, cached_raw_data = cache_manager.check_cache(
                        product_url, prod_result.markdown if use_ai else prod_result.html)

                    if is_cached and cached_raw_data:
                        if not auto_mode:
                            print(f"  [Кэш] Взят из базы: {product_url}")
                        raw_product = RawExtractedProduct(**cached_raw_data)
                    else:
                        if not auto_mode:
                            print(f"  [Извлечение] Парсинг: {product_url}")

                        if use_ai:
                            raw_product = await scraper.extract_product_data(product_url, prod_result.markdown)
                            content_to_hash = prod_result.markdown
                        else:
                            raw_product = scraper.extract_product_data(
                                product_url, prod_result.html, prod_result.markdown)
                            content_to_hash = prod_result.html

                        if str(raw_product.price_raw) == "0" or not raw_product.price_raw:
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(
                                        product_url,
                                        headers={
                                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                                        timeout=5
                                    ) as resp:
                                        if resp.status == 200:
                                            raw_html = await resp.text()
                                            nuxt_match = re.search(
                                                r'[\'"]?price[\'"]?\s*:\s*(\d+(?:\.\d+)?)', raw_html, re.IGNORECASE)
                                            if nuxt_match:
                                                raw_product.price_raw = re.sub(
                                                    r'[^\d]', '', nuxt_match.group(1))
                            except Exception:
                                pass

                        async with aiohttp.ClientSession() as session:
                            valid_images = []
                            for img_url in raw_product.images:
                                if await validate_image_url(img_url, session):
                                    valid_images.append(img_url)
                                if len(valid_images) >= 5:
                                    break
                            raw_product.images = valid_images

                        cache_manager.update_cache(
                            product_url, content_to_hash, raw_product.model_dump())

                    cat_name_clean = transformer.clean_emojis_and_specials(
                        raw_product.category_name)[:56]
                    cat_id = category_id_map.get(cat_name_clean)
                    if not cat_id:
                        cat_id = transformer.generate_numeric_id(cat_name_clean)[
                            :10]
                        category_id_map[cat_name_clean] = cat_id

                    if cat_id not in collections_map:
                        clean_cat_usp = transformer.smart_truncate(
                            transformer.clean_emojis_and_specials(raw_product.category_usp), 81)
                        collections_map[cat_id] = CategoryCollection(
                            category_id=cat_id,
                            name=cat_name_clean,
                            url=parent_category_url,
                            picture=raw_product.images[0] if raw_product.images else "",
                            description=clean_cat_usp
                        )
                    elif not collections_map[cat_id].picture and raw_product.images:
                        collections_map[cat_id].picture = raw_product.images[0]

                    transformed_product = transformer.transform(
                        raw_product, product_url, category_id_map)

                    if transformed_product.price != "0" or not skip_empty_price:
                        transformed_products.append(transformed_product)
                        if not auto_mode:
                            price_str = f"{transformed_product.price} {transformed_product.currency}" if transformed_product.price != "0" else "Без цены"
                            print(
                                f"  ✅ Успех: {transformed_product.name} ({price_str})")
                    else:
                        if not auto_mode:
                            print(
                                f"  ⏭️ Пропущен (нет цены): {transformed_product.name}")
                except Exception as e:
                    print(f"  ❌ Ошибка обработки товара: {e}")

        async with AsyncWebCrawler() as crawler:
            if direct_urls_mode:
                if not auto_mode:
                    print(
                        f"\n📦 [РЕ-ПАРСИНГ] Прямое сканирование {len(url_queue)} товаров из фида...")
                tasks = [process_single_product(
                    u, u, crawler) for u in url_queue]
                await asyncio.gather(*tasks)
            else:
                while url_queue:
                    current_url = url_queue.pop(0)
                    if current_url in visited_urls:
                        continue
                    visited_urls.add(current_url)

                    if not auto_mode:
                        print(
                            f"\n{'='*60}\n🌍 [Разведка] Анализ URL: {current_url}\n{'='*60}")

                    result = await safe_fetch(crawler, current_url, delay=1.5)
                    if not result or not result.html:
                        if not auto_mode:
                            print(
                                f"  ❌ Пропуск: Не удалось загрузить категорию {current_url}")
                        continue

                    grouped_links = discovery_agent.analyze_and_group_links(
                        current_url, result.html)
                    if not grouped_links:
                        await process_single_product(current_url, current_url, crawler)
                        continue

                    if auto_mode or not interactive_selection:
                        product_items = []
                        group_keys = list(grouped_links.keys())

                        if not interactive_selection and not auto_mode:
                            if group_keys:
                                sig = group_keys[0]
                                chosen_prod_sigs.add(sig)
                                product_items.extend(grouped_links[sig])
                        else:
                            for sig, links in grouped_links.items():
                                if sig in chosen_prod_sigs:
                                    product_items.extend(links)
                                elif sig in chosen_cat_sigs:
                                    for item in links:
                                        if item['url'] not in visited_urls and item['url'] not in url_queue:
                                            url_queue.append(item['url'])

                        if product_items:
                            tasks = [process_single_product(
                                item['url'], current_url, crawler) for item in product_items]
                            await asyncio.gather(*tasks)
                    else:
                        print("\n📦 АНАЛИЗ СТРУКТУРЫ САЙТА:")
                        group_keys = list(grouped_links.keys())
                        for idx, sig in enumerate(group_keys, 1):
                            sample_title = grouped_links[sig][0]['title'][:
                                                                          60] if grouped_links[sig] else ""
                            marker = "🟢 ТОВАРЫ/УСЛУГИ" if any(
                                x in sig for x in ['product', 'item', 'detail']) else "📁 КАТЕГОРИИ"
                            print(
                                f"  [{idx}] {marker} | {sig}\n      └─ Пример: «{sample_title}» (Ссылок: {len(grouped_links[sig])})")

                        prod_input = input(
                            "\nГруппы с ТОВАРАМИ (номера через запятую, Enter - пропустить): ").strip()
                        if prod_input:
                            selected_prod = [int(i.strip()) for i in prod_input.split(
                                ',') if i.strip().isdigit()]
                            product_items = []
                            for idx in selected_prod:
                                if 1 <= idx <= len(group_keys):
                                    sig = group_keys[idx-1]
                                    chosen_prod_sigs.add(sig)
                                    product_items.extend(grouped_links[sig])

                            if product_items:
                                limit_input = input(
                                    f"Спарсить все {len(product_items)} товаров? [Enter - Все | Число - Лимит]: ").strip()
                                limit = int(limit_input) if limit_input.isdigit() else len(
                                    product_items)
                                print(f"🚀 Запуск {limit} задач...")
                                tasks = [process_single_product(
                                    item['url'], current_url, crawler) for item in product_items[:limit]]
                                await asyncio.gather(*tasks)

                        cat_input = input(
                            "\nГруппы с КАТЕГОРИЯМИ для обхода (номера через запятую): ").strip()
                        if cat_input:
                            selected_cat = [int(i.strip()) for i in cat_input.split(
                                ',') if i.strip().isdigit()]
                            added = 0
                            for idx in selected_cat:
                                if 1 <= idx <= len(group_keys):
                                    sig = group_keys[idx-1]
                                    chosen_cat_sigs.add(sig)
                                    for item in grouped_links[sig]:
                                        if item['url'] not in visited_urls and item['url'] not in url_queue:
                                            url_queue.append(item['url'])
                                            added += 1
                            print(f"🔀 Добавлено {added} категорий в очередь.")

        all_cached_urls = cache_manager.get_all_cached_urls()
        base_prefix = urls[0] if urls and not direct_urls_mode else "https://"
        missing_urls = {
            u for u in all_cached_urls if u not in crawled_product_urls and u.startswith(base_prefix)}

        if missing_urls:
            if not auto_mode:
                print(
                    f"\n🔍 Инкрементальное обновление: найдено {len(missing_urls)} пропавших товаров. Переводим в архив.")
            for m_url in missing_urls:
                raw_data_dict = cache_manager.get_raw_data(m_url)
                if raw_data_dict:
                    raw_product = RawExtractedProduct(**raw_data_dict)
                    raw_product.available = False

                    cat_name_clean = transformer.clean_emojis_and_specials(
                        raw_product.category_name)[:56]
                    cat_id = category_id_map.get(
                        cat_name_clean, transformer.generate_numeric_id(cat_name_clean)[:10])

                    transformed_product = transformer.transform(
                        raw_product, m_url, category_id_map)
                    transformed_products.append(transformed_product)
                    if not auto_mode:
                        print(
                            f"  🔻 Переведен в архив (false): {transformed_product.name}")

        if transformed_products:
            if not auto_mode:
                print("\n📝 Генерация YML-фида...")

            if not direct_urls_mode:
                config["auto_product_signatures"] = list(chosen_prod_sigs)
                config["auto_category_signatures"] = list(chosen_cat_sigs)
                save_config(config)

            builder = YMLBuilder(
                config, datetime.now().strftime("%Y-%m-%d %H:%M"))
            builder.build_feed(transformed_products, list(
                collections_map.values()), output_filename)

            if not auto_mode:
                print(
                    f"🎉 Фид успешно сохранен: {output_filename} (Собрано оферов: {len(transformed_products)})")

                if config.get("auto_github_export", False):
                    export_dir = build_github_snapshot(
                        config, output_filename, target_url=urls[0] if not direct_urls_mode else None)
                    auto_create_and_push_github(
                        config, export_dir, target_url=urls[0] if not direct_urls_mode else None)
                else:
                    if not override_filename:
                        gh = input(
                            "\nЖелаете настроить автообновление этого фида в GitHub Actions? (y/n, Enter=Да): ").strip().lower()
                        if gh in ['y', 'yes', 'д', 'да', '']:
                            export_dir = build_github_snapshot(
                                config, output_filename, target_url=urls[0] if not direct_urls_mode else None)
                            auto_create_and_push_github(
                                config, export_dir, target_url=urls[0] if not direct_urls_mode else None)
        else:
            if not auto_mode:
                print("\n⚠️ Товары не собраны.")

    if not auto_mode and not override_filename:
        input("\nНажмите Enter, чтобы вернуться в меню...")


def main_menu():
    print("\n💡 Важно: Файл feed_cache.json необходимо удалить, чтобы скрипт спарсил новые данные.")
    config = load_config()
    while True:
        clear_screen()

        use_ai = config.get("use_ai_extraction", False)

        if use_ai:
            mode_display = "AI (DEEPSEEK)"
            key_status = "✅ Готов" if config.get(
                "deepseek_api_key", "") else "❌ НЕ НАСТРОЕН"
        else:
            mode_display = "КЛАССИКА (Эвристика + Сжатие текста)"
            key_status = "✅ Не требуется"

        cdata_mode = "AUTO" if config.get("auto_cdata", True) else "OFF"
        skip_empty = "ПРОПУСКАТЬ" if config.get(
            "skip_empty_price", True) else "СОХРАНЯТЬ"
        auto_oldprice = "ВКЛ" if config.get("auto_oldprice", True) else "ВЫКЛ"
        title_prefix = config.get("title_prefix", "")
        prefix_display = f"['{title_prefix}']" if title_prefix else "[ВЫКЛ]"
        split_feeds = "ВКЛ" if config.get("split_feeds", False) else "ВЫКЛ"
        interactive_selection = "ВКЛ" if config.get(
            "interactive_selection", True) else "ВЫКЛ"

        def_sales = config.get("default_sales_notes", "")
        sales_display = f"['{def_sales}']" if def_sales else "[Авто-поиск]"

        def_desc = config.get("default_offer_description", "")
        desc_display = f"['{def_desc[:15]}...']" if def_desc else "[Авто-поиск]"

        print("="*50)
        print("🤖 ГЕНЕРАТОР YML-ФИДОВ (AI AUTONOMIC SCRAPER 2026)")
        print("="*50)
        print(f"1. Режим извлечения: {mode_display} [{key_status}]")
        print(
            f"2. Управление списком URL (Стартовых: {len(config['target_urls'])})")
        print("3. Настройки Магазина (Имя, Компания, URL)")
        print("-" * 50)
        print(f"4. Режим CDATA (Только для description): [{cdata_mode}]")
        print(f"5. Товары без цены (0 руб): [{skip_empty}]")
        print(f"6. Авто-скидка (oldprice +10%): [{auto_oldprice}]")
        print(f"7. Префикс заголовков (Купить/Заказать): {prefix_display}")
        print(f"8. УТП по умолчанию (sales_notes): {sales_display}")
        print(f"9. Описание по умолчанию (description): {desc_display}")
        print("-" * 50)
        print(f"10. Разделять фиды (1 URL = 1 Файл): [{split_feeds}]")
        print(
            f"11. Ручной выбор паттернов (Интерактив): [{interactive_selection}]")
        print("12. 🐙 Управление GitHub (Авто-Создание Репозиториев)")
        print("13. 📝 Редактор сгенерированных фидов (YML Editor)")
        print("14. ▶ ЗАПУСТИТЬ ПАРСЕР И ГЕНЕРАЦИЮ")
        print("0. Выход")
        print("="*50)

        choice = input("Выберите действие: ").strip()

        if choice == '1':
            print("\n1. Использовать AI-анализ (Дороже, но умнее)")
            print("2. Использовать Классический парсер (Бесплатно, быстро)")
            m_choice = input("Выбор: ").strip()

            if m_choice == '1':
                config["use_ai_extraction"] = True
                cur_key = config.get("deepseek_api_key", "")
                print(
                    f"\nТекущий ключ: {cur_key[:8]}...{cur_key[-4:] if len(cur_key)>12 else 'Пусто'}")
                new_key = input(
                    "Новый API Key DeepSeek (Enter оставить текущий): ").strip()
                if new_key:
                    config["deepseek_api_key"] = new_key
                save_config(config)
            elif m_choice == '2':
                config["use_ai_extraction"] = False
                save_config(config)

        elif choice == '2':
            action = input(
                "\n[A] Добавить URL, [C] Очистить: ").strip().lower()
            if action == 'a':
                while True:
                    url = input("URL (Enter завершить): ").strip()
                    if not url:
                        break
                    config["target_urls"].append(url)
            elif action == 'c':
                config["target_urls"] = []
            save_config(config)

        elif choice == '3':
            shop = input(
                f"Название магазина [{config['shop_name']}]: ").strip()
            if shop:
                config["shop_name"] = shop
            comp = input(f"Юр. лицо [{config['company_name']}]: ").strip()
            if comp:
                config["company_name"] = comp
            site = input(f"URL сайта [{config['site_url']}]: ").strip()
            if site:
                config["site_url"] = site
            save_config(config)

        elif choice == '4':
            config["auto_cdata"] = not config.get("auto_cdata", True)
            save_config(config)

        elif choice == '5':
            config["skip_empty_price"] = not config.get(
                "skip_empty_price", True)
            save_config(config)

        elif choice == '6':
            config["auto_oldprice"] = not config.get("auto_oldprice", True)
            save_config(config)

        elif choice == '7':
            prefix = input(
                "\nВведите префикс (например: Купить) или оставьте пустым для отключения: ").strip()
            config["title_prefix"] = prefix
            save_config(config)

        elif choice == '8':
            sales = input(
                "\nВведите универсальное УТП для sales_notes (или оставьте пустым для отключения): ").strip()
            config["default_sales_notes"] = sales
            save_config(config)

        elif choice == '9':
            desc = input(
                "\nВведите общее описание для товаров (оставьте пустым для авто-поиска): ").strip()
            config["default_offer_description"] = desc
            if desc:
                use_for_coll = input(
                    "Использовать его же для коллекций? (Лимит 81 символ) (y/n, Enter=Да): ").strip().lower()
                if use_for_coll in ['y', 'yes', 'д', 'да', '']:
                    config["default_collection_description"] = DataTransformer.smart_truncate(
                        desc, 81)
                else:
                    coll_desc = input(
                        "Введите описание для коллекций (макс 81 символ): ").strip()
                    config["default_collection_description"] = DataTransformer.smart_truncate(
                        coll_desc, 81)
            else:
                config["default_collection_description"] = ""
            save_config(config)

        elif choice == '10':
            config["split_feeds"] = not config.get("split_feeds", False)
            save_config(config)

        elif choice == '11':
            config["interactive_selection"] = not config.get(
                "interactive_selection", True)
            save_config(config)

        elif choice == '12':
            while True:
                clear_screen()
                print("="*50)
                print("🐙 НАСТРОЙКИ GITHUB (АВТОМАТИЗАЦИЯ)")
                print("="*50)

                cur_pat = config.get("github_pat", "")
                pat_display = f"{cur_pat[:4]}...{cur_pat[-4:]}" if len(
                    cur_pat) > 10 else "НЕ ЗАДАН"

                print(
                    f"1. Задать GitHub Personal Access Token (Текущий: {pat_display})")

                auto_gh = "ВКЛ" if config.get(
                    "auto_github_export", False) else "ВЫКЛ"
                print(
                    f"2. Автоматическое создание репозиториев без вопросов: [{auto_gh}]")

                print("3. Просмотреть локальные папки экспорта и залить вручную")
                print("4. Проверить статус и редактировать загруженные фиды (CRON, УТП)")
                print("0. Назад")

                gh_choice = input("\nВыбор: ").strip()
                if gh_choice == '1':
                    print("\n💡 Как получить PAT: GitHub -> Settings -> Developer Settings -> Personal access tokens (classic) -> Generate new token (отметьте 'repo')")
                    pat = input("Введите токен (ghp_...): ").strip()
                    if pat:
                        config["github_pat"] = pat
                        save_config(config)
                elif gh_choice == '2':
                    config["auto_github_export"] = not config.get(
                        "auto_github_export", False)
                    save_config(config)
                elif gh_choice == '3':
                    export_dirs = [d for d in os.listdir(
                        '.') if os.path.isdir(d) and d.startswith('export_')]
                    if not export_dirs:
                        print("❌ Папки с экспортом не найдены.")
                        input("Нажмите Enter...")
                        continue

                    print("\n📦 Найденные папки для отправки в GitHub:")
                    for i, d in enumerate(export_dirs, 1):
                        print(f"[{i}] {d}")

                    try:
                        folder_choice = int(
                            input("\nВыберите номер папки (или 0 для отмены): ").strip())
                        if folder_choice == 0:
                            continue
                        target_dir = export_dirs[folder_choice - 1]

                        local_config_path = os.path.join(
                            target_dir, "feed_settings.json")
                        try:
                            with open(local_config_path, 'r', encoding='utf-8') as f:
                                local_config = json.load(f)
                                t_url = local_config.get(
                                    "target_urls", ["https://example.com"])[0]
                        except:
                            t_url = "https://example.com"

                        auto_create_and_push_github(
                            config, target_dir, target_url=t_url)
                        input("\nНажмите Enter...")
                    except (ValueError, IndexError):
                        continue
                elif gh_choice == '4':
                    export_dirs = [d for d in os.listdir(
                        '.') if os.path.isdir(d) and d.startswith('export_')]
                    if not export_dirs:
                        print("❌ Экспорты не найдены.")
                        input("Нажмите Enter...")
                        continue

                    print(
                        "\n🔍 Проверка статуса фидов на GitHub (это займет пару секунд)...")
                    try:
                        import requests
                        for i, d in enumerate(export_dirs, 1):
                            conf_path = os.path.join(d, "feed_settings.json")
                            if os.path.exists(conf_path):
                                try:
                                    with open(conf_path, 'r', encoding='utf-8') as f:
                                        lc = json.load(f)
                                    raw_link = lc.get("github_raw_url")
                                    if raw_link:
                                        resp = requests.head(raw_link)
                                        if resp.status_code == 200:
                                            print(
                                                f"✅ [{i}] ДОСТУПЕН ({d}):\n   👉 {raw_link}")
                                        else:
                                            print(
                                                f"❌ [{i}] НЕДОСТУПЕН (Код {resp.status_code}) ({d}):\n   👉 {raw_link}")
                                except Exception:
                                    pass

                        try:
                            edit_choice = int(input(
                                "\nВыберите номер фида для редактирования настроек (или 0 для отмены): ").strip())
                            if edit_choice > 0 and edit_choice <= len(export_dirs):
                                target_dir = export_dirs[edit_choice - 1]
                                edit_github_feed_settings(config, target_dir)
                        except (ValueError, IndexError):
                            pass

                    except ImportError:
                        print(
                            "❌ Установите библиотеку requests: pip install requests")
                        input("Нажмите Enter...")

                elif gh_choice == '0':
                    break

        elif choice == '13':
            YMLEditor.edit_feed_interactive(config)

        elif choice == '14':
            mode = input(
                "\nЧто формируем в фиде?\n1. ОФФЕРЫ + КОЛЛЕКЦИИ\n2. Только ОФФЕРЫ\n3. Только КОЛЛЕКЦИИ (товары как коллекции)\nВыбор: ").strip()
            if mode in ['1', '2', '3']:
                config["feed_mode"] = mode
                if mode == '1':
                    dup = input(
                        "Дублировать каждый оффер в виде коллекции для ЕПК? (y/n, Enter=Да): ").strip().lower()
                    config["duplicate_offers"] = dup in [
                        'y', 'yes', 'д', 'да', '']
                else:
                    config["duplicate_offers"] = False

                save_config(config)
                asyncio.run(run_parser(config))

        elif choice == '0':
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomic YML Generator 2026")
    parser.add_argument('--auto', action='store_true',
                        help='Run in completely headless auto mode using cached settings')
    args = parser.parse_args()

    if args.auto:
        config = load_config()
        print("⚡ Запуск в АВТОМАТИЧЕСКОМ режиме (headless)...")
        asyncio.run(run_parser(config, auto_mode=True))
    else:
        try:
            main_menu()
        except KeyboardInterrupt:
            print("\nРабота прервана пользователем.")
