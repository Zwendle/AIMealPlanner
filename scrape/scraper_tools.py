"""
Reusable web scraping utilities
"""

import time
import json
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options

# Configuration
MAX_RETRIES = 3
REQUEST_DELAY = 0.8
PRODUCT_DELAY = 0.8


def create_session(headers: Optional[Dict] = None) -> requests.Session:
    """Create a requests session with default headers"""
    session = requests.Session()
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if headers:
        default_headers.update(headers)
    session.headers.update(default_headers)
    return session


def create_driver(headless: bool = True) -> webdriver.Chrome:
    """Create a Selenium Chrome driver with optimized settings"""
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Performance optimizations
    prefs = {
        "profile.default_content_setting_values": {
            "images": 2,  # Block images for faster loading
            "plugins": 2,
            "popups": 2,
            "geolocation": 2,
            "notifications": 2,
            "media_stream": 2,
        }
    }
    options.add_experimental_option("prefs", prefs)

    try:
        from webdriver_manager.chrome import ChromeDriverManager

        return webdriver.Chrome(ChromeDriverManager().install(), options=options)
    except:
        # Fallback if webdriver_manager not available
        return webdriver.Chrome(options=options)


def get_with_retries(
    session: requests.Session, url: str, **kwargs
) -> requests.Response:
    """Make a GET request with retry logic"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30, **kwargs)
            if r.status_code in (429, 503):
                time.sleep(2 * attempt)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(1.5 * attempt)
    raise RuntimeError("Retries exhausted")


def extract_next_data_json(html_text: str) -> Optional[Dict]:
    """
    Extract __NEXT_DATA__ JSON from Next.js sites
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # Look for __NEXT_DATA__ script tag
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        try:
            return json.loads(next_data.string)
        except json.JSONDecodeError:
            pass

    # Search other script tags for JSON data
    for script in soup.find_all("script", type="application/json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    return None


def safe_find_text(element, selector: str, attribute: str = None) -> str:
    """Safely find and extract text from an element"""
    if not element:
        return ""

    found = element
    if selector:
        found = element.find(selector) if hasattr(element, "find") else None

    if not found:
        return ""

    if attribute:
        return found.get(attribute, "")

    return found.get_text(strip=True) if hasattr(found, "get_text") else str(found)


def wait_and_click(
    driver: webdriver.Chrome, selector: str, by: By = By.CSS_SELECTOR, timeout: int = 10
) -> bool:
    """Wait for element and click it"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, selector))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        time.sleep(0.5)
        element.click()
        return True
    except (TimeoutException, NoSuchElementException):
        return False


def scroll_to_load(
    driver: webdriver.Chrome, pause: float = 2.0, max_scrolls: int = 10
) -> None:
    """Scroll page to load dynamic content"""
    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0

    while scrolls < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break

        last_height = new_height
        scrolls += 1


def extract_json_ld(html_text: str, schema_type: str = None) -> Optional[Dict]:
    """Extract JSON-LD structured data from HTML"""
    soup = BeautifulSoup(html_text, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        try:
            data = json.loads(script.string)
            if schema_type:
                if isinstance(data, dict) and data.get("@type") == schema_type:
                    return data
                elif isinstance(data, list):
                    for item in data:
                        if item.get("@type") == schema_type:
                            return item
            else:
                return data
        except json.JSONDecodeError:
            continue

    return None


def extract_nutrition_from_table(soup: BeautifulSoup) -> Dict:
    """Extract nutrition information from HTML table"""
    nutrition = {}

    # Look for nutrition table
    nutrition_table = (
        soup.find("table", class_="nutrition")
        or soup.find("div", class_="nutrition-facts")
        or soup.find("div", id="nutrition")
    )

    if not nutrition_table:
        return nutrition

    # Extract rows
    rows = nutrition_table.find_all("tr") or nutrition_table.find_all(
        "div", class_="row"
    )

    for row in rows:
        # Try to find label and value
        label = row.find("td", class_="label") or row.find("span", class_="label")
        value = row.find("td", class_="value") or row.find("span", class_="value")

        if label and value:
            key = label.get_text(strip=True).lower().replace(" ", "_")
            val = value.get_text(strip=True)
            nutrition[key] = val

    return nutrition


def normalize_price(price_str: str) -> str:
    """Normalize price string to standard format"""
    if not price_str:
        return ""

    # Remove currency symbols and extra text
    price = re.sub(r"[^\d.,]", "", price_str)

    # Handle different decimal separators
    price = price.replace(",", ".")

    # Ensure it starts with $
    if price and not price.startswith("$"):
        price = f"${price}"

    return price


def extract_id_from_url(url: str) -> str:
    """Extract product ID from URL"""
    if not url:
        return ""

    # Common patterns for product IDs in URLs
    patterns = [
        r"/product/([^/]+)",
        r"/p/([^/]+)",
        r"/item/([^/]+)",
        r"/sku/([^/]+)",
        r"-(\d+)(?:\.|$|/)",
        r"id=([^&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # Fallback: use last path segment
    path = urlparse(url).path.rstrip("/")
    if path:
        return path.split("/")[-1]

    return ""


def safe_json_dumps(obj: Any) -> str:
    """Safely convert object to JSON string"""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"
