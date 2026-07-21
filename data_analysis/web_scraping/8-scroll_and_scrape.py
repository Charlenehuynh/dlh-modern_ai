#!/usr/bin/env python3
"""function that scrolls and extracts all products from infinite page"""

import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """Return a list of unique product dicts, e.g."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920, 1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(scroll_pause)
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            (driver.execute_script("window.scrollTo(0, document.body.scrollHeight);"))
            time.sleep(scroll_pause)

            str = "return document.body.scrollHeight"
            new_height = driver.execute_script(str)

            if new_height == last_height:
                break
            last_height = new_height
        cards = driver.find_elements("css selector", "div.thumbnail")

        seen = set()
        products = []

        for card in cards:
            try:
                title_el = card.find_element("css selector", "a.title")
                title = title_el.get_attribute("title")
            except Exception:
                title = None

            try:
                price = card.find_element("css selector", "h4.price").text
            except Exception:
                price = None

            try:
                description = card.find_element("css selector", "p.description").text
            except Exception:
                description = None

            try:
                stars = card.find_elements(
                    "css selector", ".ratings p.ws-icon.ws-icon-star"
                )
                rating = len(stars)
            except Exception:
                rating = 0

            key = (title, price)
            if key in seen:
                continue
            seen.add(key)

            products.append(
                {
                    "title": title,
                    "price": price,
                    "description": description,
                    "rating": rating,
                }
            )

        return products

    finally:
        driver.quit()
