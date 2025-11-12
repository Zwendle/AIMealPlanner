"""
Trader Joe's Product Scraper with Selenium
Correctly fetches all product links from pagination
and then visits each link to scrape Name, Price, and Nutrition.
"""

import csv
import time
import json
from typing import Dict, List, Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

# Import our reusable tools from parent directory
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the reusable tools you provided
from scraper_tools import (
    create_driver,
    scroll_to_load,
    safe_json_dumps,
    REQUEST_DELAY,
    PRODUCT_DELAY,
)


class TraderJoesScraper:
    def __init__(self, headless: bool = False):
        """Initialize scraper with Selenium driver"""
        self.base_url = "https://www.traderjoes.com"
        # Use the create_driver function from your tools
        self.driver = create_driver(headless=headless)
        self.driver.implicitly_wait(5)
        self.products = []
        self.seen_product_urls = set()

    def __del__(self):
        """Clean up driver on deletion"""
        if hasattr(self, "driver"):
            self.driver.quit()

    def get_categories(self) -> List[Dict]:
        """Define categories to scrape (excluding Produce)"""
        print("Getting categories (skipping Fresh Fruits & Veggies)...")
        return [
            # {
            #     "url": "https://www.traderjoes.com/home/products/category/bakery-11",
            #     "name": "Bakery",
            # },
            # {"url": "https://www.traderjoes.com/home/products/category/fresh-fruits-vegetables-flowers-31069", "name": "Produce"}, # SKIPPED
            {
                "url": "https://www.traderjoes.com/home/products/category/meat-seafood-plant-based-122",
                "name": "Meat & Seafood",
            },
            {
                "url": "https://www.traderjoes.com/home/products/category/dairy-eggs-44",
                "name": "Dairy & Eggs",
            },
            {
                "url": "https://www.traderjoes.com/home/products/category/for-the-pantry-137",
                "name": "Pantry",
            },
        ]

    def handle_initial_popups(self):
        """Handle any initial popups, cookies, or location settings"""
        print("Navigating to site and handling popups...")
        try:
            self.driver.get(f"{self.base_url}/home")
            wait = WebDriverWait(self.driver, 10)

            # 1. Handle Cookie Banner
            try:
                cookie_button = wait.until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//button[contains(text(), 'Got It')]")
                    )
                )
                self.driver.execute_script("arguments[0].click();", cookie_button)
                print("✓ Clicked 'Got It' cookie button.")
                time.sleep(0.5)
            except Exception:
                print("Note: Could not find/click cookie button.")

            # 2. Handle "Join The Adventure" Email Popup
            try:
                close_button = wait.until(
                    EC.element_to_be_clickable(
                        (
                            By.CSS_SELECTOR,
                            'div[role="dialog"] button[aria-label="Close"], button[class*="Modal_closeButton"]',
                        )
                    )
                )
                self.driver.execute_script("arguments[0].click();", close_button)
                print("✓ Closed email popup.")
                time.sleep(0.5)
            except Exception:
                print("Note: Could not find/click email popup close.")

        except Exception as e:
            print(f"Warning: Initial setup encountered an error: {e}")

    def get_product_links_from_category(self, category_url: str) -> List[str]:
        """
        Scrapes a category (and all its pages) to find all
        unique product detail page (PDP) links.
        """
        print(f"\nPHASE 1: Finding all product links in: {category_url.split('/')[-1]}")
        self.driver.get(category_url)

        try:
            # Wait for products to load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'a[href*="/home/products/pdp/"]')
                )
            )
        except TimeoutException:
            print(f"  ⚠ Timeout waiting for products in category.")
            return []

        links_found = set()
        page_num = 1

        while True:
            print(f"  Scraping page {page_num} for links...")
            # Use the scroll_to_load function from your tools
            scroll_to_load(self.driver, pause=1.5, max_scrolls=3)

            # Find all visible product links
            try:
                # This finds all links that have the product detail page URL structure
                # This is the selector you found: a.Link_link__1AZfr.ProductCard_card__title__301JH...
                product_links = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    'a[class*="ProductCard_card__title"][href*="/home/products/pdp/"]',
                )

                new_links_on_page = 0
                for link_el in product_links:
                    try:
                        href = link_el.get_attribute("href")
                        if href and href.startswith(self.base_url):
                            if href not in links_found:
                                links_found.add(href)
                                new_links_on_page += 1
                    except Exception:
                        continue  # Stale element

                print(f"    Found {new_links_on_page} new product links on this page.")
                if new_links_on_page == 0 and page_num > 1:
                    print("    No new links found, assuming end of category.")
                    break

            except Exception as e:
                print(f"    Error finding links on page: {e}")

            # Try to click next page
            try:
                next_button = self.driver.find_element(
                    By.CSS_SELECTOR,
                    'button[class*="Pagination_pagination__arrow_side_right"]',
                )
                if next_button.is_displayed() and next_button.is_enabled():
                    # Check if disabled
                    if (
                        next_button.get_attribute("disabled")
                        or next_button.get_attribute("aria-disabled") == "true"
                    ):
                        print("  Next button is disabled. End of category.")
                        break

                    self.driver.execute_script("arguments[0].click();", next_button)
                    page_num += 1
                    time.sleep(REQUEST_DELAY)  # Use delay from your tools
                else:
                    print("  No visible next button. End of category.")
                    break
            except NoSuchElementException:
                print("  No next button found. End of category.")
                break

        return list(links_found)

    def parse_product_page(
        self, product_url: str, category_name: str
    ) -> Optional[Dict]:
        """
        Visits a single product page and scrapes all details:
        Name, Price, and full Nutrition Info.
        """
        try:
            self.driver.get(product_url)
            # Wait for the main product title to be present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'h1[class*="ProductDetails_main__title"]')
                )
            )
            time.sleep(0.5)  # Let JS render

            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            product_data = {
                "category": category_name,
                "name": "",
                "price": "",
                "url": product_url,
                "id": product_url.split("/")[-1].split("-")[-1],  # Get ID from URL
                "nutrition_json": "",
            }

            # 1. Get Name
            name_el = soup.find(
                "h1", class_=lambda c: c and "ProductDetails_main__title" in c
            )
            if name_el:
                product_data["name"] = name_el.get_text(strip=True)

            # 2. Get Price
            price_el = soup.find(
                "span", class_=lambda c: c and "ProductPrice_productPrice__price" in c
            )
            if price_el:
                product_data["price"] = price_el.get_text(strip=True).replace("$", "")

            # 3. Get Nutrition (The complex part)
            nutrition = {}

            # Try to click "Per container" tab
            try:
                per_container_button = self.driver.find_element(
                    By.XPATH,
                    "//button[contains(text(), 'Per container') and not(contains(@class, 'Nav_active'))]",
                )
                if per_container_button:
                    self.driver.execute_script(
                        "arguments[0].click();", per_container_button
                    )
                    time.sleep(1)
                    # Re-parse soup after click
                    soup = BeautifulSoup(self.driver.page_source, "html.parser")
            except:
                pass  # Already active or not present

            # Find the active nutrition table
            nutrition_wrapper = soup.find(
                "div", class_=lambda c: c and "NutritionFacts_wrapper" in c
            )
            if nutrition_wrapper:
                visible_item = None
                items = nutrition_wrapper.find_all(
                    "div", class_=lambda c: c and "Item_item" in c
                )
                for item in items:
                    parent = item.parent
                    if (
                        parent
                        and parent.get("style")
                        and "block" in parent.get("style")
                    ):
                        visible_item = item
                        break
                if not visible_item and items:
                    visible_item = items[-1]  # Default to last

                if visible_item:
                    # Get Serving Size/Calories
                    chars = visible_item.find(
                        "div", class_=lambda c: c and "Item_characteristics" in c
                    )
                    if chars:
                        for item_div in chars.find_all(
                            "div",
                            class_=lambda c: c and "Item_characteristics__item" in c,
                        ):
                            title = item_div.find(
                                "div",
                                class_=lambda c: c
                                and "Item_characteristics__title" in c,
                            )
                            text = item_div.find(
                                "div",
                                class_=lambda c: c
                                and "Item_characteristics__text" in c,
                            )
                            if title and text:
                                title_text = title.get_text(strip=True).lower()
                                if "serving size" in title_text:
                                    nutrition["serving_size"] = text.get_text(
                                        strip=True
                                    )
                                elif "calories" in title_text:
                                    nutrition["calories"] = text.get_text(strip=True)

                    # Get Nutrition Table
                    table = visible_item.find(
                        "table", class_=lambda c: c and "Item_table" in c
                    )
                    if table:
                        for row in table.find_all(
                            "tr", class_=lambda c: c and "Item_table__row" in c
                        ):
                            cells = row.find_all(
                                ["td", "th"],
                                class_=lambda c: c and "Item_table__cell" in c,
                            )
                            if len(cells) >= 2:
                                nutrient = cells[0].get_text(strip=True)
                                amount = cells[1].get_text(strip=True)
                                if nutrient and amount and nutrient.lower() != "amount":
                                    key = (
                                        nutrient.lower()
                                        .replace(" ", "_")
                                        .replace("includes", "includes_added_sugars")
                                    )
                                    nutrition[key] = amount
                                    if len(cells) >= 3:
                                        dv = cells[2].get_text(strip=True)
                                        if dv and dv != "%dv":
                                            nutrition[f"{key}_dv"] = dv

            # 4. Get Ingredients
            try:
                ing_header = soup.find(
                    ["h2", "h3"], string=lambda t: t and "Ingredients" in t
                )
                if ing_header:
                    ing_text_el = ing_header.find_next_sibling(["div", "p"])
                    if ing_text_el:
                        nutrition["ingredients"] = ing_text_el.get_text(strip=True)
            except Exception:
                pass  # No ingredients found

            # Use the safe_json_dumps from your tools
            product_data["nutrition_json"] = (
                safe_json_dumps(nutrition) if nutrition else ""
            )

            return product_data

        except Exception as e:
            print(f"  ⚠ FAILED to scrape {product_url}: {e}")
            return None

    def save_to_csv(self, filename: str = "trader_joes_ALL2_PRODUCTS.csv"):
        """Save scraped products to CSV file"""
        if not self.products:
            print("No products to save")
            return

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["category", "name", "price", "url", "id", "nutrition_json"]
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self.products)

        print(f"\n✓ Saved {len(self.products)} products to {filename}")

    def run(self):
        """Main execution method"""

        self.handle_initial_popups()

        categories = self.get_categories()
        total_links = 0

        for category in categories:
            # 1. Get all product links for the category
            product_links = self.get_product_links_from_category(category["url"])
            print(f"  Found {len(product_links)} unique links for {category['name']}.")
            total_links += len(product_links)

            # 2. Scrape each product page
            print(
                f"\nPHASE 2: Now scraping {len(product_links)} product pages for {category['name']}..."
            )
            for i, link in enumerate(product_links):
                if link in self.seen_product_urls:
                    print(
                        f"  [{i + 1}/{len(product_links)}] Skipping duplicate: {link.split('/')[-1]}"
                    )
                    continue

                print(
                    f"  [{i + 1}/{len(product_links)}] Scraping: {link.split('/')[-1]}"
                )
                product_data = self.parse_product_page(link, category["name"])

                if product_data:
                    self.products.append(product_data)
                    self.seen_product_urls.add(link)

                # Use product delay from your tools
                time.sleep(PRODUCT_DELAY)

            # Save intermediate results after each category
            print(f"\nFinished {category['name']}. Saving intermediate results...")
            self.save_to_csv()

        print("\n=== Scraping Complete ===")
        print(f"  Total categories scraped: {len(categories)}")
        print(f"  Total unique products found: {len(self.products)}")
        self.save_to_csv()


def main():
    """Main entry point"""
    print("Trader Joe's Full Product Scraper")
    print("=" * 40)

    # Set headless=False to see the browser in action
    scraper = TraderJoesScraper(headless=True)  # Set to True for speed, False to watch

    try:
        scraper.run()
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"\nAn critical error occurred: {e}")
    finally:
        print("Saving final data...")
        # Save on exit/error
        scraper.save_to_csv()
        del scraper


if __name__ == "__main__":
    main()
