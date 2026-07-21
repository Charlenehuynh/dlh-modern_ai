#!/usr/bin/env python3
"""Module to scroll and scrape products from an infinite-scroll page."""

import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """Scrolls to the bottom of a page and extracts unique products."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)

    scraped_products = []
    seen_identifiers = set()

    try:
        driver.get(url)

        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            start_time = time.time()
            while time.time() - start_time < scroll_pause:
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height > last_height:
                    break
                time.sleep(0.1)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        product_cards = driver.find_elements("css selector", ".thumbnail")

        for card in product_cards:
            try:
                title_elem = card.find_element("css selector", "a.title")
                title = title_elem.get_attribute("title") or title_elem.text.strip()
            except Exception:
                title = ""

            try:
                price_elem = card.find_element("css selector", "h4.price")
                price = price_elem.text.strip()
            except Exception:
                price = ""

            try:
                desc_elem = card.find_element("css selector", "p.description")
                description = desc_elem.text.strip()
            except Exception:
                description = ""

            stars = card.find_elements("css selector", ".ratings .ws-icon-star")
            rating = len(stars)

            product_identifier = (title, price)
            if title and product_identifier not in seen_identifiers:
                seen_identifiers.add(product_identifier)
                scraped_products.append(
                    {
                        "title": title,
                        "price": price,
                        "description": description,
                        "rating": rating,
                    }
                )

    finally:
        driver.quit()

    return scraped_products
