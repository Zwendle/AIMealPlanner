import time
import json
import csv
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List

# chatgpt used to generate function comments and helped guide me through extracting next.js data

# URLs to scrape from + store ID constants
STORE_ID = "10031"
CATEGORY_URLS = [
    "https://www.wholefoodsmarket.com/products/produce",
    "https://www.wholefoodsmarket.com/products/dairy-eggs",
    "https://www.wholefoodsmarket.com/products/meat",
    "https://www.wholefoodsmarket.com/products/pantry-essentials",
    "https://www.wholefoodsmarket.com/products/breads-rolls-bakery",
    "https://www.wholefoodsmarket.com/products/desserts",
    "https://www.wholefoodsmarket.com/products/supplements",
    "https://www.wholefoodsmarket.com/products/frozen-foods",
    "https://www.wholefoodsmarket.com/products/snacks-chips-salsas-dips",
    "https://www.wholefoodsmarket.com/products/seafood",
    "https://www.wholefoodsmarket.com/products/beverages",
]

BASE = "https://www.wholefoodsmarket.com"
CATEGORY_API = "https://www.wholefoodsmarket.com/api/products/category/{slug}"

# api request constants
LIMIT = 60
REQUEST_DELAY = 0.8
PRODUCT_DELAY = 0.8
MAX_RETRIES = 3

# session setup
session = requests.Session()
session.headers.update({
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/140.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.wholefoodsmarket.com/products",
})

def get_slug_from_category_url(url: str) -> str:
    """
    Extracts the category slug from a category URL.
    Args:
        url (str): The category URL to extract the slug from
    
    Returns:
        str: The last path segment after /products/, or empty string if not found
    """
    path = urlparse(url).path.strip("/").split("/")
    if not path:
        return ""
    return path[-1]

def get_with_retries(url, **kwargs):
    """
    Performs an HTTP GET request with retry logic and exponential backoff.
    
    Args:
        url (str): The URL to request
        **kwargs (Any): Additional arguments to pass to requests.get()
    
    Returns:
        requests.Response: The response object from a successful request
    
    Raises:
        RuntimeError: If all retry attempts are exhausted
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30, **kwargs)
            if r.status_code in (429, 503):
                # backoff on rate limiting
                time.sleep(2 * attempt)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(1.5 * attempt)
    raise RuntimeError("Unreachable: retries exhausted")

def ensure_cookies_for_domain():
    """
    Makes a priming request to allow the site to set cookies/session for store/locale association.
    
    Args:
        None
    
    Returns:
        None
    """
    try:
        _ = get_with_retries(f"{BASE}/products/produce")
    except Exception:
        pass

def extract_products_from_category_json(data):
    """
    Extracts product list from category API JSON response, handling various key structures.
    
    Args:
        data (Dict[str, Any]): The JSON response dictionary from the category API
    
    Returns:
        List[Dict[str, Any]]: A list of product dictionaries, or empty list if none found
    """
    # guessing common keys
    for key in ("products", "items", "results", "data", "page", "payload"):
        val = data.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            return val
        if isinstance(val, dict):
            for subkey in ("products", "items", "results"):
                subval = val.get(subkey)
                if isinstance(subval, list) and subval and isinstance(subval[0], dict):
                    return subval

    for v in data.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            if any(("name" in v[0], "label" in v[0])):
                return v
    return []

def id_from_slug(slug: str):
    """
    Extracts product ID from a slug string.
    
    Args:
        slug (str): The product slug string
    
    Returns:
        Optional[str]: The last hyphen-separated chunk if it's 5-20 characters, otherwise None
    """
    if not slug:
        return None
    tail = slug.rsplit("-", 1)[-1]
    return tail if 5 <= len(tail) <= 20 else None

def normalize_product_fields(p, category_slug):
    """
    Normalizes product dictionary to a standard format with required fields.
    
    Args:
        p (Dict[str, Any]): Raw product dictionary from API
        category_slug (str): The category slug for this product
    
    Returns:
        Dict[str, Any]: Dictionary with normalized fields: name, price, url, id, category, store, raw
    """
    name = p.get("name")
    price = p.get("regularPrice")
    slug = p.get("slug")
    url = urljoin("https://www.wholefoodsmarket.com", f"/product/{slug}") if slug else None
    pid = id_from_slug(slug)
    store = p.get("store")

    return {
        "name": name,
        "price": price,
        "url": url,
        "id": pid,
        "category": category_slug,
        "store": store,
        "raw": p
    }

def paginate_category(category_url, store_id):
    """
    Fetches all products from a category by paginating through API results.
    
    Args:
        category_url (str): The category page URL
        store_id (str): The store ID to filter products by
    
    Returns:
        List[Dict[str, Any]]: List of normalized product dictionaries from the category
    """
    slug = get_slug_from_category_url(category_url)
    api_url = CATEGORY_API.format(slug=slug)
    offset = 0
    all_products = []

    while True:
        params = {
            "leafCategory": slug,
            "store": store_id,
            "limit": str(LIMIT),
            "offset": str(offset),
        }
        r = get_with_retries(api_url, params=params)
        data = r.json()

        page_items = extract_products_from_category_json(data)
        if not page_items:
            break

        for p in page_items:
            all_products.append(normalize_product_fields(p, slug))

        # stop conditions
        if len(page_items) < LIMIT or data.get("hasMore") is False:
            break

        offset += LIMIT
        time.sleep(REQUEST_DELAY)

    return all_products

def extract_next_data_json(html_text):
    """
    Extracts JSON data from Next.js page's __NEXT_DATA__ script tag.
    
    Args:
        html_text (str): The HTML content of the page
    
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dictionary from __NEXT_DATA__ script, or None if not found
    """
    soup = BeautifulSoup(html_text, "html.parser")

    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        try:
            return json.loads(next_data.string)
        except json.JSONDecodeError:
            pass

    # search for pageProps in other script tags
    for script in soup.find_all("script"):
        if not script.string:
            continue
        txt = script.string.strip()
        if "pageProps" in txt and txt.startswith("{") and txt.endswith("}"):
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                continue

    return None

def find_keys_containing(obj, substring, path=""):
    """
    Recursively searches a nested dict/list for keys containing a substring.
    
    Args:
        obj (Any): The nested dictionary or list to search
        substring (str): The substring to search for in keys
        path (str): Current path prefix for nested structures (used internally)
    
    Returns:
        List[tuple]: List of (path, value) tuples where path is like 'nutritionPanel.servingSizeDisplay'
    """
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            if substring.lower() in k.lower():
                results.append((new_path, v))
            results.extend(find_keys_containing(v, substring, new_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            results.extend(find_keys_containing(v, substring, new_path))
    return results

def extract_nutrition_data(next_data: dict) -> Optional[dict]:
    """
    Extracts nutrition data from Next.js data object structure.
    
    Args:
        next_data (Dict[str, Any]): The parsed __NEXT_DATA__ JSON dictionary
    
    Returns:
        Optional[Dict[str, Any]]: Nutrition data dictionary from props.pageProps.data, or None if not found
    """
    if not isinstance(next_data, dict):
        return None

    props = next_data.get("props") or {}
    pageProps = props.get("pageProps") or next_data.get("pageProps") or {}
    data = pageProps.get("data") or {}

    return data if isinstance(data, dict) else None


def build_compact_nutrition(data: dict) -> Dict[str, Any]:
    """
    Builds a compact nutrition dictionary from raw nutrition data.
    
    Args:
        data (Dict[str, Any]): Dictionary containing nutritionElements and servingInfo
    
    Returns:
        Dict[str, Any]: Compact dictionary with serving size and key nutrition values per serving
    """
    elements = data.get("nutritionElements") or []
    by_key = {
        e.get("key"): e
        for e in elements
        if isinstance(e, dict) and "key" in e
    }
    print(f"Nutrition elements found: {list(by_key.keys())}")

    def val(key: str):
        e = by_key.get(key)
        if not e:
            return None
        # grab perServing if possible, else fall back to display
        return e.get("perServing", e.get("perServingDisplay"))

    def unit(key: str):
        e = by_key.get(key)
        if not e:
            return None
        return e.get("uom")

    serving_info = data.get("servingInfo") or {}
    serving_size = None
    if serving_info:
        size = serving_info.get("servingSize")
        uom = serving_info.get("servingSizeUom")
        if size is not None and uom:
            serving_size = f"{size} {uom}"
        else:
            # if display field exists, can fall back to it
            serving_size = serving_info.get("servingSizeDisplay")

    servings_per_container = serving_info.get("servingsPerContainerDisplay")

    compact = {
        "serving_size": serving_size,
        "servings_per_container": servings_per_container,

        # grab net weight just in case
        "total_package_size": serving_info.get("totalSize"),
        "total_package_size_uom": serving_info.get("totalSizeUom"),
        "total_package_size_secondary": serving_info.get("secondaryTotalSize"),
        "total_package_size_secondary_uom": serving_info.get("secondaryTotalSizeUom"),

        # essential nutrition values
        "calories": val("calories"),

        "total_fat": {
            "amount": val("totalFat"),
            "unit": unit("totalFat"),
        },
        "saturated_fat": {
            "amount": val("saturatedFat"),
            "unit": unit("saturatedFat"),
        },
        "trans_fat": {
            "amount": val("transFat"),
            "unit": unit("transFat"),
        },

        "cholesterol": {
            "amount": val("cholesterol"),
            "unit": unit("cholesterol"),
        },
        "sodium": {
            "amount": val("sodium"),
            "unit": unit("sodium"),
        },

        "carbohydrates": {
            "amount": val("carbohydrates"),
            "unit": unit("carbohydrates"),
        },
        "fiber": {
            "amount": val("fiber"),
            "unit": unit("fiber"),
        },
        "sugars": {
            "amount": val("sugar"),
            "unit": unit("sugar"),
        },

        "protein": {
            "amount": val("protein"),
            "unit": unit("protein"),
        },
    }

    return compact

def fetch_product_page_and_nutrition(url: str) -> Optional[dict]:
    """
    Fetches product page HTML and extracts nutrition information.
    
    Args:
        url (str): The product page URL
    
    Returns:
        Optional[Dict[str, Any]]: Compact nutrition dictionary, or None if extraction fails
    """
    if not url:
        return None

    r = get_with_retries(url)
    nd = extract_next_data_json(r.text)
    if not nd:
        return None

    data = extract_nutrition_data(nd)
    if not data:
        return None
    
    matches = find_keys_containing(data, "serv")
    for path, val in matches:
        pass

    return build_compact_nutrition(data)


def main():
    """
    Main scraper function that processes all categories and writes results to CSV.
    """
    ensure_cookies_for_domain()

    rows = []
    seen_urls = set()

    for cat_url in CATEGORY_URLS:
        print(f"== Category: {cat_url}")
        products = paginate_category(cat_url, STORE_ID)
        print(f"  -> Found {len(products)} products")

        for idx, prod in enumerate(products, 1):
            url = prod["url"]
            if not url or url in seen_urls:
                # skip if no detail URL or already processed
                row = {
                    "category": prod["category"],
                    "name": prod["name"],
                    "price": prod["price"],
                    "url": url or "",
                    "id": prod["id"] or "",
                    "nutrition_json": ""
                }
                rows.append(row)
                continue

            nutrition = None
            try:
                nutrition = fetch_product_page_and_nutrition(url)
            except Exception:
                nutrition = None

            row = {
                "category": prod["category"],
                "name": prod["name"],
                "price": prod["price"],
                "url": url,
                "id": prod["id"] or "",
                "nutrition_json": json.dumps(nutrition, ensure_ascii=False) if nutrition else ""
            }
            rows.append(row)
            seen_urls.add(url)

            if idx % 10 == 0:
                print(f"    Processed {idx}/{len(products)} in {prod['category']}...")
            time.sleep(PRODUCT_DELAY)

        # brief pause between categories
        time.sleep(1.5)

    # write CSV
    out_csv = "wholefoods_inventory_store_{sid}.csv".format(sid=STORE_ID)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category","name","price","url","id","nutrition_json"])
        w.writeheader()
        w.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    # feel free to run the scraper on a test URL! Just uncomment the lines below and comment out main()
    # test_url = "https://www.wholefoodsmarket.com/product/365-by-whole-foods-market-organic-baby-spinach-16-oz-b074h55njk"

    # result = fetch_product_page_and_nutrition(test_url)
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    main()
