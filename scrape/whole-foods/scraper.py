import time
import json
import csv
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any

# ----------------------------
# CONFIG
# ----------------------------
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

LIMIT = 60
REQUEST_DELAY = 0.8
PRODUCT_DELAY = 0.8
MAX_RETRIES = 3

# ----------------------------
# SESSION SETUP
# ----------------------------
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
    # last path segment after /products/
    path = urlparse(url).path.strip("/").split("/")
    print(f"get slug cat path: {path}")
    if not path:
        return ""
    # Expected ["products", "<slug>"]
    return path[-1]

def get_with_retries(url, **kwargs):
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
    # Shouldn't reach here
    raise RuntimeError("Unreachable: retries exhausted")

def ensure_cookies_for_domain():
    """
    Priming request to let the site set any cookies/session it needs,
    including store/locale association.
    """
    try:
        _ = get_with_retries(f"{BASE}/products/produce")
        # If you know cookie names (e.g., 'storeId'), you can also set them explicitly:
        # session.cookies.set("storeId", STORE_ID, domain=".wholefoodsmarket.com")
    except Exception:
        pass

def extract_products_from_category_json(data):
    """
    The category API may use different keys. Try common ones,
    else try to sniff a list of dicts that look like products.
    """
    # Common keys guess
    for key in ("products", "items", "results", "data", "page", "payload"):
        val = data.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            return val
        if isinstance(val, dict):
            # Sometimes nested
            for subkey in ("products", "items", "results"):
                subval = val.get(subkey)
                if isinstance(subval, list) and subval and isinstance(subval[0], dict):
                    return subval

    # Heuristic: any top-level list of dicts with a 'name' or 'label'
    for v in data.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            if any(("name" in v[0], "label" in v[0])):
                return v
    return []

def id_from_slug(slug: str):
    # grab last hyphen-separated chunk (often an ASIN-like id)
    if not slug:
        return None
    tail = slug.rsplit("-", 1)[-1]
    return tail if 5 <= len(tail) <= 20 else None  # loose guard

def normalize_product_fields(p, category_slug):
    """
    Map product dict to {name, price, url, id, category}
    Adjust paths based on real keys you observe.
    """
    name = p.get("name")
    price = p.get("regularPrice")  # <- from your sample
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

        # # DEBUG: inspect the shape once per category
        # if offset == 0:  # only print for the first page of each category
        #     print("Top-level keys in category JSON:", list(data.keys())[:15])
        #     print("First item keys:", list(page_items[0].keys())[:25])

        #     import json, pathlib
        #     pathlib.Path("debug").mkdir(exist_ok=True)
        #     with open(f"debug/sample_{slug}.json", "w", encoding="utf-8") as f:
        #         json.dump(
        #             {
        #                 "data_keys": list(data.keys()),
        #                 "first_item": page_items[0],
        #             },
        #             f,
        #             ensure_ascii=False,
        #             indent=2,
        #         )

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
    On Next.js sites, a script tag often contains the data:
    <script id="__NEXT_DATA__" type="application/json"> ... </script>
    """
    soup = BeautifulSoup(html_text, "html.parser")
    # Prefer id="__NEXT_DATA__"
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        try:
            return json.loads(next_data.string)
        except json.JSONDecodeError:
            pass

    # Sometimes there are other script tags with JSON-LD or chunks that include 'pageProps'
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
    Recursively search a nested dict/list for keys that contain `substring`.
    Returns a list of (path, value) pairs, where path is like
    'nutritionPanel.servingSizeDisplay'.
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
    Pull the nutrition-related payload from the Next.js data object:
    usually props.pageProps.data
    """
    if not isinstance(next_data, dict):
        return None

    props = next_data.get("props") or {}
    pageProps = props.get("pageProps") or next_data.get("pageProps") or {}
    data = pageProps.get("data") or {}

    return data if isinstance(data, dict) else None


def build_compact_nutrition(data: dict) -> Dict[str, Any]:
    """
    Turn the big nutritionElements list into a small JSON with just
    serving size + key per-serving numbers.
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
        # prefer numeric perServing, fall back to display if needed
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
            # if they ever add a display field, you can fall back to it here
            serving_size = serving_info.get("servingSizeDisplay")

    servings_per_container = serving_info.get("servingsPerContainerDisplay")

    compact = {
        # headline serving info
        "serving_size": serving_size,
        "servings_per_container": servings_per_container,

        # you also have these if you ever want net weight:
        "total_package_size": serving_info.get("totalSize"),
        "total_package_size_uom": serving_info.get("totalSizeUom"),
        "total_package_size_secondary": serving_info.get("secondaryTotalSize"),
        "total_package_size_secondary_uom": serving_info.get("secondaryTotalSizeUom"),

        # core label numbers (per serving)
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
    if not url:
        return None

    r = get_with_retries(url)
    nd = extract_next_data_json(r.text)
    if not nd:
        return None

    data = extract_nutrition_data(nd)
    if not data:
        return None
    
    # DEBUG: see where "serving" fields live
    matches = find_keys_containing(data, "serv")
    for path, val in matches:
        pass
        # print(path, "=>", val)

    return build_compact_nutrition(data)


def main():
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
                # Skip if no detail URL or already processed
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

    # Write CSV
    out_csv = "wholefoods_inventory_store_{sid}.csv".format(sid=STORE_ID)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category","name","price","url","id","nutrition_json"])
        w.writeheader()
        w.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    # # test new nutrition facts scraping
    # test_url = "https://www.wholefoodsmarket.com/product/365-by-whole-foods-market-organic-baby-spinach-16-oz-b074h55njk"

    # result = fetch_product_page_and_nutrition(test_url)
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    main()
