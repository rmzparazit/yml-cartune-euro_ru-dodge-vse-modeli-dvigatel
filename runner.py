import os, re, sys, time, asyncio, hashlib, json, aiohttp, backoff
import importlib
import inspect
from io import BytesIO
from PIL import Image
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any
from lxml import etree
from crawl4ai import AsyncWebCrawler
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# --- ВШИТЫЙ ПАТТЕРН ---


class Pattern:
    def match_score(self, url: str, html_content: str, raw_product) -> float:
        text_lower = html_content.lower()
        score = 0.0
        if any(w in text_lower for w in ["б/у", "пробег", "контрактн", "двигатель", "акпп", "мкпп", "cartune-euro.ru"]): score += 0.9
        return min(score, 1.0)
        
    def get_custom_labels(self, html_content: str, brand: str, cat_name: str) -> list[str]:
        custom_labels = []
        if brand and brand != "Unknown": custom_labels.append(brand)
        if cat_name and cat_name != "Каталог": custom_labels.append(cat_name)
        page_text_lower = html_content.lower()
        if any(w in page_text_lower for w in ["б/у", "бывш", "пробег"]): custom_labels.append("Б/У")
        if "контрактн" in page_text_lower: custom_labels.append("Контрактный")
        return custom_labels[:5]

    def clean_description(self, text: str) -> str:
        fluff_patterns = [
            r'(?i)комментарий от продавца[:\-\s]*', r'(?i)внимание[:\-\s]*', r'(?i)уважаемые покупатели[:\-\s]*',
            r'(?i)мы предоставляем полный пакет документов.*?учёт[:\-\s]*', r'(?i)копия грузовой.*?деклараци[ии].*?(?=\.|\n|$)',
            r'(?i)договор купли-продажи.*?(?=\.|\n|$)', r'(?i)есть аукционный лист.*?(?=\.|\n|$)',
            r'(?i)предоставим подробное фото.*?видео.*?(?=\.|\n|$)', r'(?i)возможна проверка эндоскопом.*?(?=\.|\n|$)',
            r'(?i)цена указана за.*?фото.*?(?=\.|\n|$)', r'(?i)описание товара[:\-\s]*',
            r'(?i)возможна продажа без навесного.*?([.\n]|$)', r'(?i)возможна продажа.*?([.\n]|$)',
            r'(?i)(Номер по производителю|Производитель|Марка|Модель|Год|Кузов|Артикул)[\s:]*$' 
        ]
        for pattern in fluff_patterns: 
            text = re.sub(pattern, '', text)
        return text

    def generate_keywords(self, type_prefix: str, specs: dict) -> str:
        stop_keys = ['марка', 'бренд', 'производитель', 'модель']
        safe_specs = [str(v) for k, v in specs.items() if str(k).lower() not in stop_keys]
        specs_str = " ".join(safe_specs)
        return f"{type_prefix} товар {specs_str}".strip()

    # 💡 НОВЫЙ МЕТОД ДЛЯ КРАСИВЫХ И КОРОТКИХ ОПИСАНИЙ КОЛЛЕКЦИЙ ИЗ ОФФЕРОВ
    def generate_offer_collection_desc(self, raw_product) -> str:
        """
        Генерирует описание коллекции на основе оффера, 
        используя тип, марку, модель и основные характеристики.
        Вместо обрезки H1, мы собираем текст с нуля.
        """
        type_prefix = raw_product.h1_title.split()[0] if raw_product.h1_title else "Запчасть"
        brand = raw_product.brand if raw_product.brand != "Unknown" else ""
        
        # Пытаемся вытащить модель из specs
        model = raw_product.specs.get('Модель', '')
        if not model:
            # Если модели в specs нет, берем из H1, убирая тип и бренд
            model = raw_product.h1_title.replace(type_prefix, '', 1).strip()
            if brand:
                model = re.sub(rf'(?i){re.escape(brand)}\b', '', model).strip()
        
        # Добавляем кузов и двигатель, если они есть
        extra_info = []
        if 'Кузов' in raw_product.specs: extra_info.append(f"кузов {raw_product.specs['Кузов']}")
        if 'Двигатель' in raw_product.specs: extra_info.append(f"двс {raw_product.specs['Двигатель']}")
        
        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
        
        # Собираем строку
        desc = f"Контрактная {type_prefix} {brand} {model}{extra_str}"
        
        # Чистим от двойных пробелов
        desc = re.sub(r'\s+', ' ', desc).strip()
        
        # Если получилось длиннее 81 символа, обрезаем аккуратно (без потери смысла)
        if len(desc) > 81:
            desc = desc[:78]
            desc = desc.rsplit(' ', 1)[0] + '...'
            
        return desc
# ----------------------

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
    variations: list[dict] = Field(default_factory=list)
    ai_templates: dict = Field(default_factory=dict)
    selectors: dict = Field(default_factory=dict)
    semantic_pattern: str = Field(default="")
    collection_description: str = Field(default="")
    extraction_mode: str = Field(default="classic")

    model_config = ConfigDict(populate_by_name=True)


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
    specs: dict[str, Any]
    custom_labels: list[str]
    available: str
    category_id: str
    collection_description: str = ""


class CategoryCollection(BaseModel):
    category_id: str
    name: str
    url: str
    picture: str
    description: str

# ==========================================
# СИСТЕМА ДИНАМИЧЕСКИХ ПАТТЕРНОВ (ENSEMBLE)
# ==========================================


class PatternManager:
    def __init__(self, patterns_dir="patterns"):
        self.patterns = []
        if not os.path.exists(patterns_dir): return
        
        for filename in os.listdir(patterns_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f"patterns.{module_name}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and hasattr(attr, 'match_score'):
                            self.patterns.append({"name": module_name, "obj": attr()})
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки паттерна {filename}: {e}")

    def apply_best_patterns(self, url: str, html_content: str, raw_product: RawExtractedProduct) -> tuple[RawExtractedProduct, str]:
        if not self.patterns: return raw_product, ""

        scored_patterns = []
        for p_dict in self.patterns:
            try:
                score = p_dict["obj"].match_score(url, html_content, raw_product)
                if score > 0.1: scored_patterns.append((score, p_dict))
            except: pass

        if not scored_patterns: return raw_product, ""

        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        top_patterns = [p_dict["obj"] for score, p_dict in scored_patterns[:2]]
        best_pattern_name = scored_patterns[0][1]["name"]

        for p in top_patterns:
            if hasattr(p, 'fix_price'):
                raw_product = p.fix_price(raw_product, html_content)
            if hasattr(p, 'clean_title'):
                raw_product.h1_title = p.clean_title(raw_product.h1_title)
            if hasattr(p, 'clean_description'):
                raw_product.description_usp = p.clean_description(raw_product.description_usp)
            if hasattr(p, 'generate_offer_description'):
                raw_product.ai_templates['custom_offer_desc'] = p.generate_offer_description(raw_product)
            if hasattr(p, 'filter_specs'):
                raw_product.specs = p.filter_specs(raw_product.specs)
            if hasattr(p, 'get_dynamic_category'):
                cat_name = p.get_dynamic_category(raw_product)
                if cat_name: raw_product.category_name = cat_name
            if hasattr(p, 'generate_offer_collection_desc'):
                # 💡 Умный анализатор сигнатуры для локального скрипта
                sig_params = list(inspect.signature(p.generate_offer_collection_desc).parameters.keys())
                if 'raw_product' in sig_params:
                    raw_product.collection_description = p.generate_offer_collection_desc(raw_product)
                else:
                    raw_product.collection_description = p.generate_offer_collection_desc(raw_product.h1_title)

        labels_set = set(raw_product.custom_labels)
        for p in top_patterns:
            if hasattr(p, 'get_custom_labels'):
                new_labels = p.get_custom_labels(html_content, raw_product.brand, raw_product.category_name)
                labels_set.update(new_labels)
        
        raw_product.custom_labels = list(labels_set)
        
        for p in top_patterns:
            if hasattr(p, 'filter_labels'):
                raw_product.custom_labels = p.filter_labels(raw_product.custom_labels, raw_product.h1_title)

        raw_product.custom_labels = raw_product.custom_labels[:5]
        return raw_product, best_pattern_name

# ==========================================
# УТИЛИТА: ВАЛИДАЦИЯ ИЗОБРАЖЕНИЙ
# ==========================================

class CacheManager:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f: return json.load(f)
            except: pass
        return {}

    def save(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=4, ensure_ascii=False)

    def generate_fingerprint(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'lxml')
        h1_el = soup.find(['h1', 'h2'], class_=re.compile(r'title|name|head', re.I)) or soup.find('h1')
        h1_title = h1_el.get_text(strip=True) if h1_el else "unknown_item"

        price_val = "0"
        price_regex = r'(?<!\d)(\d[\d\s.,\xa0]{0,15})\s*(₽|руб|rub|usd|€|eur|₸|kzt|byn)'
        for node in soup.find_all(class_=re.compile(r'price|cost|amount', re.I)):
            matches = re.findall(price_regex, node.get_text(separator=' ').lower(), re.IGNORECASE)
            for val_str, _ in matches:
                clean_val = re.sub(r'[^\d]', '', val_str)
                if clean_val and int(clean_val) > 0:
                    price_val = clean_val
                    break
            if price_val != "0": break

        page_text_lower = html_content.lower()
        available = "false" if any(w in page_text_lower for w in ["нет в наличии", "out of stock", "под заказ"]) else "true"
        return hashlib.md5(f"{h1_title}_{price_val}_{available}".encode('utf-8')).hexdigest()

    def check_cache(self, url: str, html_content: str) -> tuple[bool, Optional[dict]]:
        if not html_content:
            return (True, self.cache[url].get("raw_data")) if url in self.cache else (False, None)
        current_fingerprint = self.generate_fingerprint(html_content)
        if url in self.cache and self.cache[url].get("fingerprint") == current_fingerprint:
            return True, self.cache[url].get("raw_data")
        return False, None

    def update_cache(self, url: str, html_content: str, raw_data: dict):
        self.cache[url] = {"fingerprint": self.generate_fingerprint(html_content), "raw_data": raw_data, "last_seen": datetime.now().isoformat()}
        self.save()

    def patch_cache(self, url: str, new_price: str, new_oldprice: str, is_available: bool):
        if url in self.cache:
            self.cache[url]["raw_data"].update({"price": new_price, "oldprice": new_oldprice, "available": is_available})
            self.cache[url]["last_seen"] = datetime.now().isoformat()
            self.save()
            return True
        return False

    def get_all_cached_urls(self): return set(self.cache.keys())
    def get_raw_data(self, url: str): return self.cache.get(url, {}).get("raw_data")

    def get_few_shot_examples(self, domain: str, limit: int = 2) -> list:
        examples = []
        for url, data in self.cache.items():
            if domain in url and "raw_data" in data and data["raw_data"].get("h1_title") not in [None, "Без названия"]:
                rd = data["raw_data"]
                examples.append({"name": rd["h1_title"], "description": rd.get("description_usp"), "sales_notes": rd.get("sales_notes")})
                if len(examples) >= limit: break
        return examples

# ==========================================
# 3. ФАЗА РАЗВЕДКИ (DISCOVERY)
# ==========================================

class DiscoveryAgent:
    @staticmethod
    def analyze_and_group_links(base_url: str, html_content: str) -> dict:
        groups = {}
        base_parsed = urlparse(base_url)
        soup = BeautifulSoup(html_content, 'lxml')
        stop_words = ['login', 'cart', 'korzina', 'tel:', 'mailto:', '.jpg', '.png', 'policy', 'consent', 'contacts', 'kontakt', 'pro-o-nas', 'about', 'oplata', 'dostavka', 'rezerv', 'vozvrat', 'otzyvy', 'faq', 'help', 'garantiya', '+7', '8800', 'javascript:', 'whatsapp', 'viber', 'tg://', 'auth', 'register']

        for a in soup.find_all('a', href=True):
            href = a.get('href')
            parsed_href = urlparse(href)
            if (parsed_href.netloc and parsed_href.netloc != base_parsed.netloc) or any(w in parsed_href.path.lower() for w in stop_words): continue

            clean_href = href.split('#')[0].split('?')[0]
            if clean_href in ['/', '', base_parsed.netloc]: continue

            full_url = urljoin(base_url, clean_href)
            path_parts = [p for p in parsed_href.path.lower().split('/') if p]
            signature = f"/{'/'.join(path_parts[:-1])}/*" if len(path_parts) > 1 else f"/{path_parts[0]}/*" if path_parts else "/"

            title = ""
            parent_card = a.find_parent(['div', 'li', 'article'], class_=lambda c: c and any(x in c.lower() for x in ['product', 'item', 'good', 'card']))
            if parent_card:
                texts = [t for t in parent_card.stripped_strings if t]
                if texts: title = " | ".join(texts[:3])

            if not title:
                title = a.get_text(strip=True) or a.get('title', '')
                if not title and a.find('img'): title = a.find('img').get('alt', '') or a.find('img').get('title', '')

            title = ' '.join(title.split()) if title else "Без названия"
            if signature not in groups: groups[signature] = {}
            if full_url not in groups[signature] or len(title) > len(groups[signature].get(full_url, "")): groups[signature][full_url] = title

        result = {sig: [{"url": k, "title": v} for k, v in links.items()] for sig, links in groups.items() if links}
        return dict(sorted(result.items(), key=lambda i: (1 if any(x in i[0] for x in ['product', 'item', 'detail', 'catalog/']) else 0, len(i[1])), reverse=True))

    @staticmethod
    def detect_single_page_catalog(html_content: str, domain_rules: dict) -> list[BeautifulSoup]:
        soup = BeautifulSoup(html_content, 'lxml')
        product_blocks = []

        block_selector = domain_rules.get("selectors", {}).get("product_block")
        if block_selector:
            return soup.select(block_selector)

        possible_classes = ['product-item', 'product-card', 'item', 'product', 'card', 'catalog-item', 'goods-item', 't-store__card']
        
        for class_name in possible_classes:
            found_blocks = soup.find_all(class_=re.compile(class_name, re.I))
            if found_blocks:
                for block in found_blocks:
                    text_content = block.get_text(separator=' ').lower()
                    if block.find(class_=re.compile(r'price|title|name', re.I)) and ('руб' in text_content or '₽' in text_content or '$' in text_content):
                         product_blocks.append(block)
                
                if len(product_blocks) > 0:
                     return list(set(product_blocks))

        return []

# ==========================================
# 4. ФАЗА ИЗВЛЕЧЕНИЯ (КЛАССИКА С СЕЛЕКТОРАМИ И ИИ)
# ==========================================

class ClassicScraper:
    @staticmethod
    def extract_product_data(url: str, html_content: str, markdown_content: str = "", domain_rules: dict = None) -> RawExtractedProduct:
        soup = BeautifulSoup(html_content, 'lxml')
        
        common_trash = ['form', '.popup', '.modal', '.cookie', '#popup', '#modal', '.auth-form']
        for trash_sel in common_trash:
            try:
                for el in soup.select(trash_sel): el.decompose()
            except: pass
            
        if domain_rules and "exclude_from_parsing" in domain_rules:
            for bad_selector in domain_rules["exclude_from_parsing"]:
                try:
                    for el in soup.select(bad_selector): el.decompose()
                except: pass

        h1_title = ""
        price_raw = "0"
        oldprice_raw = "0"
        desc_usp = ""
        images = []
        specs = {}
        brand = "Unknown"

        selectors = domain_rules.get("selectors", {}) if domain_rules else {}
        
        if selectors:
            try:
                if selectors.get("h1_title"):
                    el = soup.select_one(selectors["h1_title"])
                    if el: h1_title = el.get_text(strip=True)
                if selectors.get("price"):
                    el = soup.select_one(selectors["price"])
                    if el:
                        clean_val = re.sub(r'[^\d]', '', el.get_text())
                        if clean_val: price_raw = clean_val
                if selectors.get("oldprice"):
                    el = soup.select_one(selectors["oldprice"])
                    if el:
                        clean_val = re.sub(r'[^\d]', '', el.get_text())
                        if clean_val: oldprice_raw = clean_val
                if selectors.get("description"):
                    el = soup.select_one(selectors["description"])
                    if el: desc_usp = el.get_text(separator=' ', strip=True)
                if selectors.get("images"):
                    img_els = soup.select(selectors["images"])
                    for img in img_els:
                        src = img.get('src') or img.get('data-src') or img.get('data-lazy')
                        if src and src.startswith('http'): images.append(src)
                        elif src and src.startswith('//'): images.append('https:' + src)
                
                if selectors.get("specs_block"):
                    for block in soup.select(selectors["specs_block"]):
                        name_el = block.select_one(selectors.get("specs_name", ".name")) if selectors.get("specs_name") else None
                        val_els = block.select(selectors.get("specs_value", ".value")) if selectors.get("specs_value") else [block]
                        
                        n_text = name_el.get_text(strip=True) if name_el else "Параметр"
                        if name_el and not selectors.get("specs_value"):
                            name_el.decompose()
                            
                        vals = []
                        for v in val_els:
                            v_txt = v.get_text(strip=True)
                            if v_txt and v_txt != n_text and v_txt not in vals: 
                                vals.append(v_txt)
                                
                        if vals:
                            specs[n_text] = vals if len(vals) > 1 else vals[0]
            except Exception as e:
                print(f"  [Ошибка селектора]: {e}")

        if not h1_title:
            h1_el = soup.find(['h1', 'h2', 'h3', 'div'], class_=re.compile(r'title|name|head', re.I)) or soup.find('h1')
            h1_title = h1_el.get_text(strip=True) if h1_el else "Без названия"

        currency = "RUB"
        currency_map = {'₽': 'RUB', 'руб': 'RUB', 'rub': 'RUB', 'р.': 'RUB', 'р': 'RUB', '$': 'USD', 'usd': 'USD', '€': 'EUR', 'eur': 'EUR', '₸': 'KZT', 'kzt': 'KZT', 'byn': 'BYN'}
        price_regex = r'(?<!\d)(\d[\d\s.,\xa0]{0,15})\s*(₽|руб|rub|usd|€|eur|₸|kzt|byn)'

        if price_raw == "0":
            def find_price() -> tuple[str, str]:
                if markdown_content:
                    clean_md = markdown_content.replace('&nbsp;', '').replace('&#160;', '')
                    matches_md = re.findall(price_regex, clean_md.lower(), re.IGNORECASE)
                    for val_str, curr_str in matches_md:
                        clean_val = re.sub(r'[^\d]', '', val_str)
                        if clean_val and int(clean_val) > 0: return clean_val, currency_map.get(curr_str.strip().lower(), 'RUB')

                for node in soup.find_all(class_=re.compile(r'price|cost|amount', re.I)):
                    text = node.get_text(separator=' ').lower()
                    matches = re.findall(price_regex, text, re.IGNORECASE)
                    for val_str, curr_str in matches:
                        clean_val = re.sub(r'[^\d]', '', val_str)
                        if clean_val and int(clean_val) > 0: return clean_val, currency_map.get(curr_str.strip().lower(), 'RUB')
                return "0", "RUB"
            price_raw, currency = find_price()

        if oldprice_raw == "0":
            for oldprice_node in soup.find_all(class_=re.compile(r'old-price|old_price|price-old|crossed', re.I)):
                clean_str = re.sub(r'[^\d]', '', oldprice_node.get_text())
                if clean_str and int(clean_str) > 0:
                    oldprice_raw = clean_str
                    break

        if not images:
            for img_container in soup.find_all(['div', 'a', 'img', 'picture'], class_=re.compile(r'img|image|slider|gallery|photo', re.I)):
                img = img_container if img_container.name == 'img' else img_container.find('img')
                if img:
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy')
                    if src and src.startswith('http'): images.append(src)
                    elif src and src.startswith('//'): images.append('https:' + src)

        heuristic_specs = {}
        for row in soup.find_all('tr'):
            cols = row.find_all(['td', 'th'])
            if len(cols) == 2: heuristic_specs[cols[0].get_text(strip=True)] = cols[1].get_text(strip=True)

        for item in soup.find_all(['li', 'div'], class_=re.compile(r'param|property|feature|attribute|options', re.I)):
            name_el = item.find(['div', 'span'], class_=re.compile(r'name|title|label', re.I))
            val_el = item.find(['div', 'span', 'ul'], class_=re.compile(r'value|val|list', re.I))
            if name_el and val_el:
                heuristic_specs[name_el.get_text(strip=True)] = val_el.get_text(separator=', ', strip=True)
            else:
                text = item.get_text(separator=' ', strip=True)
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts[0]) < 30: heuristic_specs[parts[0].strip()] = parts[1].strip()

        for k, v in heuristic_specs.items():
            if k not in specs: specs[k] = v
            if str(k).lower() in ['марка', 'производитель', 'бренд']: brand = str(v)

        if not desc_usp:
            desc_container = soup.find(['div', 'section'], class_=re.compile(r'comment|description|detail-text|about', re.I))
            if desc_container: desc_usp = desc_container.get_text(separator=' ', strip=True)

        sales_notes = ""
        delivery_nodes = soup.find_all(string=re.compile(r'(предоплат|оплат|картой|доставк|тк |отправк|гарант|возврат|срок)', re.I))
        for node in delivery_nodes:
            parent = getattr(node, 'parent', None)
            if parent and parent.name not in ['script', 'style']:
                text = re.sub(r'\s+', ' ', parent.get_text(separator=' ', strip=True))
                if re.search(r'\d+', text) and 5 < len(text) <= 50:
                    sales_notes = text
                    break

        cat_name = "Каталог"
        breadcrumbs = soup.find_all(['span', 'li', 'a', 'div'], class_=re.compile(r'breadcrumb|bx-breadcrumb|nav', re.I))
        if len(breadcrumbs) > 1:
            cat_name = breadcrumbs[-1].get_text(strip=True)
        elif not breadcrumbs and url:
            path_parts = [p for p in urlparse(url).path.split('/') if p]
            if path_parts: cat_name = path_parts[-1].replace('-', ' ').title()

        custom_labels = []
        if brand != "Unknown": custom_labels.append(brand)
        if cat_name != "Каталог": custom_labels.append(cat_name)

        page_text_lower = soup.get_text(separator=' ').lower()
        if "б/у" in page_text_lower or "бывш" in page_text_lower or "пробег" in page_text_lower:
            custom_labels.append("Б/У")
        if "контрактн" in page_text_lower:
            custom_labels.append("Контрактный")

        return RawExtractedProduct(
            h1_title=h1_title, brand=brand, price_raw=price_raw, oldprice_raw=oldprice_raw,
            currency=currency, images=list(set(images)), specs=specs, category_name=cat_name,
            category_usp=cat_name, description_usp=desc_usp, sales_notes=sales_notes, custom_labels=custom_labels[:5], variations=[], ai_templates={},
            extraction_mode="classic"
        )


class DataTransformer:
    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def clean_punctuation(text: str) -> str:
        text = str(text)
        text = re.sub(r'[,;]\s*\.', '.', text)
        text = re.sub(r'\.\s*,', '.', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'^[\s.,!?;:\-]+', '', text)
        text = re.sub(r'[\s\-.,!?;:]+$', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def smart_truncate(text: str, max_length: int, is_collection: bool = False) -> str:
        text = DataTransformer.clean_punctuation(text)
        if len(text) > max_length:
            if is_collection:
                sub_text = text[:max_length]
                last_punct = max(sub_text.rfind('.'), sub_text.rfind('!'), sub_text.rfind('?'))
                if last_punct > 15: text = sub_text[:last_punct+1]
                else: text = sub_text.rsplit(' ', 1)[0]
            else:
                text = text[:max_length].rsplit(' ', 1)[0]

        junk_pattern = r'\s+(и|в|на|с|от|до|за|по|к|из|у|без|для|про|а|но|да|или|как|что|где|когда|если)$'
        for _ in range(3):
            old_text = text
            text = re.sub(junk_pattern, '', text, flags=re.IGNORECASE).strip()
            if text == old_text: break

        text = DataTransformer.clean_punctuation(text)
        if text and is_collection: 
            text = text[0].upper() + text[1:]
            text = text.rstrip('.')
        return text

    @staticmethod
    def generate_numeric_id(text: str) -> str:
        return str(int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16))[:90]

    @staticmethod
    def clean_emojis_and_specials(text: str) -> str:
        clean_text = re.sub(r'[^\w\s.,!?\-:;/"\'()]', '', str(text))
        return re.sub(r'\s+', ' ', clean_text).strip()

    @staticmethod
    def compress_commercial_text(text: str, max_length: int = 175, is_collection: bool = False) -> str:
        if not text: return ""
        text = re.sub(r'(?i)id\s*товара\s*\d+.*$', '', text)
        fluff_patterns = [
            r'(?i)комментарий от продавца[:\-\s]*', r'(?i)внимание[:\-\s]*', r'(?i)уважаемые покупатели[:\-\s]*',
            r'(?i)мы предоставляем полный пакет документов.*?учёт[:\-\s]*', r'(?i)копия грузовой.*?деклараци[ии].*?(?=\.|\n|$)',
            r'(?i)договор купли-продажи.*?(?=\.|\n|$)', r'(?i)есть аукционный лист.*?(?=\.|\n|$)',
            r'(?i)предоставим подробное фото.*?видео.*?(?=\.|\n|$)', r'(?i)возможна проверка эндоскопом.*?(?=\.|\n|$)',
            r'(?i)цена указана за.*?фото.*?(?=\.|\n|$)', r'(?i)описание товара[:\-\s]*',
            r'(?i)возможна продажа без навесного.*?([.\n]|$)', r'(?i)возможна продажа.*?([.\n]|$)',
            r'(?i)(Номер по производителю|Производитель|Марка|Модель|Год|Кузов|Артикул)[\s:]*$'
        ]
        for pattern in fluff_patterns: text = re.sub(pattern, '', text)
        text = re.sub(r':\s*-\s*', ': ', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        return DataTransformer.smart_truncate(text, max_length, is_collection)

    @staticmethod
    def parse_universal_price(raw_price_val: Any) -> float:
        if not raw_price_val: return 0.0
        if isinstance(raw_price_val, (int, float)): return float(raw_price_val)
        clean_str = re.sub(r'[^\d.,]', '', str(raw_price_val).replace('\xa0', '').replace(' ', '')).replace(',', '.')
        parts = clean_str.split('.')
        if len(parts) > 2: clean_str = ''.join(parts[:-1]) + '.' + parts[-1]
        try: return float(clean_str) if clean_str else 0.0
        except ValueError: return 0.0

    def apply_title_prefix(self, title: str) -> str:
        prefix = self.config.get("title_prefix", "").strip()
        if not prefix: return title
        words = title.split()
        if not words: return title
        first_word = words[0]
        if not (re.search(r'[A-Za-z0-9]', first_word) or first_word.isupper()):
            words[0] = first_word.lower()
        return f"{prefix} {' '.join(words)}"

    def apply_spin_template(self, tmpl: str, context: dict) -> str:
        if not tmpl: return ""
        res = str(tmpl)
        name_val = str(context.get("name", ""))
        
        if "{name}" in res and name_val:
            words = name_val.split()
            first_word = words[0] if words else ""
            if first_word and not (re.match(r'^[A-Za-z0-9]+$', first_word) or first_word.isupper()):
                words[0] = first_word.lower()
            lowered_name = " ".join(words)
            
            if res.startswith("{name}"):
                res = res.replace("{name}", name_val, 1).replace("{name}", lowered_name)
            else:
                res = res.replace("{name}", lowered_name)
                
        for k, v in context.items():
            if k != "name":
                res = res.replace(f"{{{k}}}", str(v))
                
        return res.strip()

    def transform_multiple(self, raw: RawExtractedProduct, url: str, category_id_map: dict) -> list[TransformedProduct]:
        valid_price = self.parse_universal_price(raw.price_raw)
        if (raw.h1_title == "Без названия" or not raw.h1_title) and valid_price == 0: return []

        valid_oldprice = self.parse_universal_price(raw.oldprice_raw)
        if self.config.get("auto_oldprice", True) and valid_oldprice == 0 and valid_price > 0:
            valid_oldprice = valid_price * 1.10

        clean_h1 = self.clean_emojis_and_specials(raw.h1_title)
        words = clean_h1.split()
        type_prefix = words[0] if words else "Товар"

        vendor_clean = self.clean_emojis_and_specials(raw.brand)[:50]
        if vendor_clean.lower() == "unknown" or not vendor_clean:
            vendor_clean = self.config.get("company_name", "Не указан")

        model_str = clean_h1
        model_str = re.sub(rf'^{re.escape(type_prefix)}\s*', '', model_str, flags=re.IGNORECASE)
        model_str = re.sub(rf'{re.escape(vendor_clean)}\s*', '', model_str, flags=re.IGNORECASE).strip()
        if not model_str: model_str = raw.specs.get('Модель', 'Без модели')

        base_desc_full = self.compress_commercial_text(self.clean_emojis_and_specials(raw.description_usp), max_length=9999, is_collection=False)
        base_desc = self.smart_truncate(base_desc_full, 175, is_collection=False)
        leftover_desc = base_desc_full[len(base_desc):].strip()

        default_sales = self.config.get("default_sales_notes", "").strip()
        extracted_sales = self.clean_emojis_and_specials(raw.sales_notes).strip()

        dynamic_fallback_sales = ""
        if leftover_desc: dynamic_fallback_sales = self.smart_truncate(leftover_desc, 50, is_collection=False)
        if not dynamic_fallback_sales or len(dynamic_fallback_sales) < 5:
            valid_labels = [lbl for lbl in raw.custom_labels if lbl]
            if valid_labels: dynamic_fallback_sales = self.smart_truncate(", ".join(valid_labels), 50, is_collection=False)
        if not dynamic_fallback_sales: dynamic_fallback_sales = "Товар проверен, в наличии"

        safe_specs = {}
        stop_param_keys = ['производитель', 'бренд', 'марка', 'модель']
        stop_param_values = ['none', 'null', 'n/a', 'не указан', 'нет', '-', '', 'стандартный']

        for k, v in raw.specs.items():
            clean_k = self.clean_emojis_and_specials(str(k)).strip()
            if isinstance(v, list):
                clean_v = [self.clean_emojis_and_specials(str(i)).strip() for i in v if self.clean_emojis_and_specials(str(i)).strip().lower() not in stop_param_values]
                if clean_v: safe_specs[clean_k] = clean_v
            else:
                clean_v = self.clean_emojis_and_specials(str(v)).strip()
                if clean_k.lower() not in stop_param_keys and clean_v.lower() not in stop_param_values:
                    safe_specs[clean_k] = clean_v

        cat_name_clean = self.clean_emojis_and_specials(raw.category_name)[:56]
        if not cat_name_clean or cat_name_clean == "Без названия": cat_name_clean = "Каталог"
        cat_id = category_id_map.get(cat_name_clean, self.generate_numeric_id(cat_name_clean)[:10])

        price_str = "0" if valid_price == 0 else f"{int(valid_price)}"
        oldprice_str = "0" if valid_oldprice == 0 else f"{int(valid_oldprice)}"

        spin_enabled = self.config.get("spin_enabled", False)
        spin_templates = self.config.get("spin_templates", ["{name}"])
        def_desc_tmpl = self.config.get("default_offer_description", "").strip()
        custom_titles_enabled = self.config.get("custom_titles_enabled", False)
        custom_titles_map = self.config.get("custom_titles_map", {})
        
        base_context = {
            "price": price_str,
            "oldprice": oldprice_str,
            "vendor": vendor_clean,
            "category": cat_name_clean
        }
        
        results = []

        if custom_titles_enabled and url in custom_titles_map and custom_titles_map[url]:
            for i, custom_name in enumerate(custom_titles_map[url]):
                ctx = base_context.copy()
                ctx["name"] = clean_h1
                
                v_title_raw = self.apply_spin_template(custom_name, ctx)
                v_title = self.smart_truncate(v_title_raw, 56, is_collection=False)
                
                ctx_desc = base_context.copy()
                ctx_desc["name"] = v_title

                if raw.ai_templates.get('custom_offer_desc'):
                    v_desc_raw = self.apply_spin_template(raw.ai_templates['custom_offer_desc'], ctx_desc)
                    v_desc = self.smart_truncate(v_desc_raw, 175, is_collection=False)
                elif def_desc_tmpl:
                    merged = def_desc_tmpl.replace("{base_desc}", base_desc)
                    v_desc = self.smart_truncate(self.apply_spin_template(merged, ctx_desc), 175, is_collection=False)
                elif base_desc: v_desc = self.smart_truncate(base_desc, 175, is_collection=False)
                else: v_desc = v_title

                if default_sales: final_sales = self.smart_truncate(self.apply_spin_template(default_sales, ctx_desc), 50, is_collection=False)
                elif extracted_sales: final_sales = self.smart_truncate(extracted_sales, 50, is_collection=False)
                else: final_sales = dynamic_fallback_sales

                results.append(TransformedProduct(
                    offer_id=self.generate_numeric_id(url + str(i)), url=url, name=v_title, type_prefix=type_prefix, vendor=vendor_clean, model_name=model_str,
                    price=price_str, oldprice=oldprice_str if valid_oldprice > valid_price else "", currency=raw.currency, images=raw.images[:5], description=v_desc, sales_notes=final_sales, specs=safe_specs,
                    custom_labels=[self.clean_emojis_and_specials(lbl)[:175] for lbl in raw.custom_labels[:5]], available="true" if raw.available else "false", category_id=cat_id,
                    collection_description=raw.collection_description
                ))
                
        elif spin_enabled and spin_templates:
            for i, tmpl in enumerate(spin_templates):
                clean_ai_title = clean_h1
                var_desc = ""
                if raw.variations and i < len(raw.variations):
                    var_item = raw.variations[i]
                    if isinstance(var_item, dict):
                        clean_ai_title = self.clean_emojis_and_specials(var_item.get("title", clean_h1))
                        var_desc = var_item.get("description", "")
                    else:
                        clean_ai_title = self.clean_emojis_and_specials(str(var_item))

                ctx = base_context.copy()
                ctx["name"] = clean_ai_title
                
                v_title_raw = self.apply_spin_template(tmpl, ctx)
                v_title = self.smart_truncate(v_title_raw, 56, is_collection=False)
                
                ctx_desc = base_context.copy()
                ctx_desc["name"] = v_title

                if raw.ai_templates.get('custom_offer_desc'):
                    v_desc_raw = self.apply_spin_template(raw.ai_templates['custom_offer_desc'], ctx_desc)
                    v_desc = self.smart_truncate(v_desc_raw, 175, is_collection=False)
                elif var_desc: 
                    v_desc = self.smart_truncate(self.clean_emojis_and_specials(var_desc), 175, is_collection=False)
                elif def_desc_tmpl: 
                    merged = def_desc_tmpl.replace("{base_desc}", base_desc)
                    v_desc = self.smart_truncate(self.apply_spin_template(merged, ctx_desc), 175, is_collection=False)
                elif base_desc: 
                    v_desc = self.smart_truncate(base_desc, 175, is_collection=False)
                else: 
                    v_desc = v_title

                if default_sales: final_sales = self.smart_truncate(self.apply_spin_template(default_sales, ctx_desc), 50, is_collection=False)
                elif extracted_sales: final_sales = self.smart_truncate(extracted_sales, 50, is_collection=False)
                else: final_sales = dynamic_fallback_sales

                results.append(TransformedProduct(
                    offer_id=self.generate_numeric_id(url + str(i)), url=url, name=v_title, type_prefix=type_prefix, vendor=vendor_clean, model_name=model_str,
                    price=price_str, oldprice=oldprice_str if valid_oldprice > valid_price else "", currency=raw.currency, images=raw.images[:5], description=v_desc, sales_notes=final_sales, specs=safe_specs,
                    custom_labels=[self.clean_emojis_and_specials(lbl)[:175] for lbl in raw.custom_labels[:5]], available="true" if raw.available else "false", category_id=cat_id,
                    collection_description=raw.collection_description
                ))
        else:
            prefixed_name = self.apply_title_prefix(clean_h1)
            ctx = base_context.copy()
            ctx["name"] = prefixed_name
            
            v_title = self.smart_truncate(prefixed_name, 56, is_collection=False)
            
            ctx_desc = base_context.copy()
            ctx_desc["name"] = v_title

            if raw.ai_templates.get('custom_offer_desc'):
                v_desc_raw = self.apply_spin_template(raw.ai_templates['custom_offer_desc'], ctx_desc)
                v_desc = self.smart_truncate(v_desc_raw, 175, is_collection=False)
            elif def_desc_tmpl:
                merged = def_desc_tmpl.replace("{base_desc}", base_desc)
                v_desc = self.smart_truncate(self.apply_spin_template(merged, ctx_desc), 175, is_collection=False)
            elif base_desc: v_desc = self.smart_truncate(base_desc, 175, is_collection=False)
            else: v_desc = v_title

            if default_sales: final_sales = self.smart_truncate(self.apply_spin_template(default_sales, ctx_desc), 50, is_collection=False)
            elif extracted_sales: final_sales = self.smart_truncate(extracted_sales, 50, is_collection=False)
            else: final_sales = dynamic_fallback_sales

            results.append(TransformedProduct(
                offer_id=self.generate_numeric_id(url), url=url, name=v_title, type_prefix=type_prefix, vendor=vendor_clean, model_name=model_str,
                price=price_str, oldprice=oldprice_str if valid_oldprice > valid_price else "", currency=raw.currency, images=raw.images[:5], description=v_desc, sales_notes=final_sales, specs=safe_specs,
                custom_labels=[self.clean_emojis_and_specials(lbl)[:175] for lbl in raw.custom_labels[:5]], available="true" if raw.available else "false", category_id=cat_id,
                collection_description=raw.collection_description
            ))

        return results

# ==========================================
# 6. ФАЗА СЕРИАЛИЗАЦИИ YML
# ==========================================


class YMLBuilder:
    def __init__(self, config: dict, date_str: str):
        self.config, self.date_str = config, date_str

    def _add_element(self, parent, tag, text, is_desc=False):
        if not text: return
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(text)).strip()
        if not text: return
        
        el = etree.SubElement(parent, tag)
        if self.config.get("cdata_mode", "auto") == "all" or (is_desc and self.config.get("cdata_mode", "auto") == "auto" and re.search(r'[&<>\'"]', text)): 
            el.text = etree.CDATA(text)
        else: 
            el.text = text.replace('"', '&quot;').replace("'", '&apos;')

    def build_feed(self, products: list[TransformedProduct], collections: list[CategoryCollection], output_path: str):
        root = etree.Element('yml_catalog', date=self.date_str)
        shop = etree.SubElement(root, 'shop')
        self._add_element(shop, 'name', self.config.get("shop_name", "Shop"))
        self._add_element(shop, 'company', self.config.get("company_name", "Company"))
        
        url_text = self.config.get("site_url", "https://example.com")
        if url_text: etree.SubElement(shop, 'url').text = url_text

        currencies_el = etree.SubElement(shop, 'currencies')
        for c in sorted(list(set(p.currency for p in products) | {"RUB"})): etree.SubElement(currencies_el, 'currency', id=c, rate="1" if c == "RUB" else "CBRF")

        categories_el = etree.SubElement(shop, 'categories')
        has_categories = False
        
        for coll in collections: 
            if coll.name and coll.name != "Без названия":
                cat_el = etree.SubElement(categories_el, 'category', id=coll.category_id)
                cat_el.text = DataTransformer.smart_truncate(coll.name, 56, True)
                has_categories = True
                
        if not has_categories:
            cat_el = etree.SubElement(categories_el, 'category', id="1")
            cat_el.text = "Каталог товаров"
            for prod in products: prod.category_id = "1"

        if self.config.get("feed_mode", "1") in ['1', '2'] and products:
            offers_el = etree.SubElement(shop, 'offers')
            for prod in products:
                offer = etree.SubElement(offers_el, 'offer', id=prod.offer_id, available=prod.available, type="vendor.model")
                self._add_element(offer, 'name', prod.name)
                self._add_element(offer, 'url', prod.url)
                etree.SubElement(offer, 'price').text = prod.price
                if prod.oldprice and prod.oldprice != "0": etree.SubElement(offer, 'oldprice').text = prod.oldprice
                etree.SubElement(offer, 'currencyId').text = prod.currency
                etree.SubElement(offer, 'categoryId').text = prod.category_id
                
                for img in prod.images: self._add_element(offer, 'picture', img)
                self._add_element(offer, 'typePrefix', prod.type_prefix)
                self._add_element(offer, 'vendor', prod.vendor)
                self._add_element(offer, 'model', prod.model_name)
                self._add_element(offer, 'description', prod.description, True)
                self._add_element(offer, 'sales_notes', prod.sales_notes)
                
                for i, lbl in enumerate(prod.custom_labels): self._add_element(offer, f'custom_label_{i}', lbl)
                
                for key, val in prod.specs.items(): 
                    if isinstance(val, list):
                        for v in val:
                            if v:
                                param_el = etree.SubElement(offer, 'param', name=str(key))
                                param_el.text = str(v)
                    elif val:
                        param_el = etree.SubElement(offer, 'param', name=str(key))
                        param_el.text = str(val)

        feed_mode = self.config.get("feed_mode", "1")
        dup_offers = self.config.get("duplicate_offers", False)
        
        if feed_mode in ['1', '3'] or dup_offers:
            collections_el = etree.SubElement(shop, 'collections')
            def_desc = self.config.get("default_collection_description", "").strip()
            
            if feed_mode in ['1', '3']:
                for coll in collections:
                    cel = etree.SubElement(collections_el, 'collection', id=coll.category_id)
                    self._add_element(cel, 'url', coll.url)
                    self._add_element(cel, 'name', DataTransformer.smart_truncate(coll.name, 56, True))
                    
                    if coll.description: 
                        final_coll_desc = coll.description
                    else: 
                        final_coll_desc = def_desc.replace("{name}", coll.name)
                        
                    final_coll_desc = DataTransformer.smart_truncate(final_coll_desc, 81, True).rstrip('.')
                    
                    self._add_element(cel, 'description', final_coll_desc, True)
                    self._add_element(cel, 'picture', coll.picture)
            
            if dup_offers or feed_mode == '3':
                for prod in products:
                    cel = etree.SubElement(collections_el, 'collection', id=f"col_{prod.offer_id}")
                    self._add_element(cel, 'url', prod.url)
                    self._add_element(cel, 'name', prod.name)
                    
                    if prod.collection_description:
                        final_coll_desc = prod.collection_description
                    else:
                        c_desc = prod.description if prod.description else def_desc
                        if not c_desc: c_desc = prod.name
                        final_coll_desc = c_desc.replace("{name}", prod.name)
                        
                    final_coll_desc = DataTransformer.smart_truncate(final_coll_desc, 81, True).rstrip('.')
                    
                    self._add_element(cel, 'description', final_coll_desc, True)
                    if prod.images: self._add_element(cel, 'picture', prod.images[0])

        etree.ElementTree(root).write(output_path, pretty_print=True, xml_declaration=True, encoding='utf-8')

# ==========================================
# GITHUB ИНТЕГРАЦИЯ И СБОРЩИК RUNNER'А
# ==========================================


async def validate_image_url(url: str, session: aiohttp.ClientSession) -> bool:
    try:
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.read()
                img = Image.open(BytesIO(data))
                return img.size[0] >= 450 and img.size[1] >= 450
    except: return False
    return False

# ==========================================
# 2. УМНЫЙ КЭШ И ПРАВИЛА ДОМЕНА
# ==========================================

def load_domain_rules(url: str) -> dict:
    domain = urlparse(url).netloc.replace('www.', '').replace('.', '_')
    fname = f"rules_{domain}.json"
    if os.path.exists(fname):
        try:
            with open(fname, 'r', encoding='utf-8') as f: return json.load(f)
        except: pass
    return {}


def save_domain_rules(url: str, rules: dict):
    domain = urlparse(url).netloc.replace('www.', '').replace('.', '_')
    with open(f"rules_{domain}.json", 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=4, ensure_ascii=False)




def apply_single_pattern(url, html_content, raw_product, pattern_obj):
    if not pattern_obj: return raw_product
    
    if hasattr(pattern_obj, 'fix_price'):
        raw_product = pattern_obj.fix_price(raw_product, html_content)
    if hasattr(pattern_obj, 'clean_title'):
        raw_product.h1_title = pattern_obj.clean_title(raw_product.h1_title)
    if hasattr(pattern_obj, 'clean_description'):
        raw_product.description_usp = pattern_obj.clean_description(raw_product.description_usp)
    if hasattr(pattern_obj, 'generate_offer_description'):
        raw_product.ai_templates['custom_offer_desc'] = pattern_obj.generate_offer_description(raw_product)
    if hasattr(pattern_obj, 'filter_specs'):
        raw_product.specs = pattern_obj.filter_specs(raw_product.specs)
    if hasattr(pattern_obj, 'get_dynamic_category'):
        cat_name = pattern_obj.get_dynamic_category(raw_product)
        if cat_name: raw_product.category_name = cat_name
    if hasattr(pattern_obj, 'generate_offer_collection_desc'):
        sig_params = list(inspect.signature(pattern_obj.generate_offer_collection_desc).parameters.keys())
        if 'raw_product' in sig_params:
            raw_product.collection_description = pattern_obj.generate_offer_collection_desc(raw_product)
        else:
            raw_product.collection_description = pattern_obj.generate_offer_collection_desc(raw_product.h1_title)

    labels_set = set(raw_product.custom_labels)
    if hasattr(pattern_obj, 'get_custom_labels'):
        new_labels = pattern_obj.get_custom_labels(html_content, raw_product.brand, raw_product.category_name)
        labels_set.update(new_labels)
    
    raw_product.custom_labels = list(labels_set)
    if hasattr(pattern_obj, 'filter_labels'):
        raw_product.custom_labels = pattern_obj.filter_labels(raw_product.custom_labels, raw_product.h1_title)
        
    raw_product.custom_labels = raw_product.custom_labels[:5]
    return raw_product

async def run_github_worker():
    with open("feed_settings.json", "r", encoding="utf-8") as f: config = json.load(f)
    t_urls = config.get("target_urls", [])
    if not t_urls: return
    
    print("🚀 Запуск GitHub Worker (Classic-Mode + Вшитый паттерн)...")
    transformer = DataTransformer(config)
    discovery_agent = DiscoveryAgent()
    cache_manager = CacheManager("feed_cache.json")
    scraper = ClassicScraper()
    
    try: active_pattern = Pattern()
    except Exception: active_pattern = None
    
    skip_empty_price = config.get("skip_empty_price", True)
    output_filename = config.get("output_file", "feed.xml")
    direct_urls_mode = config.get("direct_urls_mode", False)
    use_ai = config.get("use_ai_extraction", False)
    
    transformed_products = []
    collections_map = {}
    category_id_map = {}
    domain_rules = load_domain_rules(t_urls[0])
    
    url_queue = list(t_urls)
    visited_urls = set()
    crawled_product_urls = set()
    semaphore = asyncio.Semaphore(1)
    
    async def process_single_product(product_url: str, parent_category_url: str, html_content_or_block: str = "", is_html_block: bool = False):
        async with semaphore:
            try:
                crawled_product_urls.add(product_url)
                if not is_html_block:
                    await asyncio.sleep(2.5)
                    return
                
                is_cached, cached_raw_data = cache_manager.check_cache(product_url, html_content_or_block)
                url_in_cache = product_url in cache_manager.cache
                
                if is_cached and cached_raw_data:
                    if "price_raw" in cached_raw_data: cached_raw_data["price"] = cached_raw_data.pop("price_raw")
                    if "oldprice_raw" in cached_raw_data: cached_raw_data["oldprice"] = cached_raw_data.pop("oldprice_raw")
                    raw_product = RawExtractedProduct(**cached_raw_data)
                elif url_in_cache:
                    temp_raw = scraper.extract_product_data(product_url, html_content_or_block, "", domain_rules)
                    cache_manager.patch_cache(product_url, temp_raw.price_raw, temp_raw.oldprice_raw, temp_raw.available)
                    cached_raw_data = cache_manager.get_raw_data(product_url)
                    if "price_raw" in cached_raw_data: cached_raw_data["price"] = cached_raw_data.pop("price_raw")
                    if "oldprice_raw" in cached_raw_data: cached_raw_data["oldprice"] = cached_raw_data.pop("oldprice_raw")
                    raw_product = RawExtractedProduct(**cached_raw_data)
                else:
                    raw_product = scraper.extract_product_data(product_url, html_content_or_block, "", domain_rules)
                    raw_product = apply_single_pattern(product_url, html_content_or_block, raw_product, active_pattern)

                    semantic_pattern = domain_rules.get("semantic_pattern", "")
                    if semantic_pattern and not raw_product.description_usp:
                        text = semantic_pattern.replace("{brand}", raw_product.brand).replace("{category_name}", raw_product.category_name)
                        for k, v in raw_product.specs.items(): 
                            text = text.replace("{specs[" + str(k) + "]}", str(v))
                        raw_product.description_usp = re.sub(r'\{specs\[.*?\]\}', '', text).strip()
                        
                    cache_manager.update_cache(product_url, html_content_or_block, raw_product.model_dump())
                    
                multi_products = transformer.transform_multiple(raw_product, product_url, category_id_map)
                for p in multi_products:
                    if p.category_id not in collections_map:
                        c_name = transformer.clean_emojis_and_specials(raw_product.category_name)[:56]
                        c_desc = ""
                        if active_pattern and hasattr(active_pattern, 'get_category_description'):
                            c_desc = active_pattern.get_category_description(c_name)
                        if not c_desc:
                            c_desc = transformer.smart_truncate(transformer.clean_emojis_and_specials(raw_product.category_usp), 81, is_collection=True).rstrip('.')
                        collections_map[p.category_id] = CategoryCollection(category_id=p.category_id, name=c_name, url=parent_category_url, picture=raw_product.images[0] if raw_product.images else "", description=c_desc)
                    if p.price != "0" or not skip_empty_price:
                        transformed_products.append(p)
            except Exception as e: pass

    async with AsyncWebCrawler() as crawler:
        pass 
            
    all_cached_urls = cache_manager.get_all_cached_urls()
    
    if direct_urls_mode:
        missing_urls = {u for u in all_cached_urls if u not in crawled_product_urls and u in t_urls}
    else:
        missing_urls = {u for u in all_cached_urls if u not in crawled_product_urls and u.startswith(t_urls[0])}
    
    for m_url in missing_urls:
        raw_data_dict = cache_manager.get_raw_data(m_url)
        if raw_data_dict:
            raw_product = RawExtractedProduct(**raw_data_dict)
            raw_product.available = False
            
            transformed_products_archived = transformer.transform_multiple(raw_product, m_url, category_id_map)
            for p in transformed_products_archived:
                if p.category_id not in collections_map:
                    c_name = transformer.clean_emojis_and_specials(raw_product.category_name)[:56]
                    c_desc = ""
                    if active_pattern and hasattr(active_pattern, 'get_category_description'):
                        c_desc = active_pattern.get_category_description(c_name)
                    if not c_desc:
                        c_desc = transformer.smart_truncate(transformer.clean_emojis_and_specials(raw_product.category_usp), 81, is_collection=True).rstrip('.')
                        
                    collections_map[p.category_id] = CategoryCollection(
                        category_id=p.category_id, name=c_name, url=m_url,
                        picture=raw_product.images[0] if raw_product.images else "", description=c_desc
                    )
            
            transformed_products.extend(transformed_products_archived)
            
    if transformed_products:
        builder = YMLBuilder(config, datetime.now().strftime("%Y-%m-%d %H:%M"))
        builder.build_feed(transformed_products, list(collections_map.values()), output_filename)
        print(f"🎉 Фид сохранен: {output_filename}")

if __name__ == "__main__":
    asyncio.run(run_github_worker())
