#!/usr/bin/env python3
"""function that scrolls and extracts all products from infinite page"""

import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait


def scroll_and_scrape(url, scroll_pause=2.0):
    """Return a list of unique product dicts."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.page_load_strategy = "eager"  # don't wait for every subresource

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)  # never hang forever on .get()

    try:
        driver.get(url)

        last_height = driver.execute_script("return document.body.scrollHeight")

        max_iterations = 50  # safety valve against a true infinite loop
        for _ in range(max_iterations):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Poll instead of blindly sleeping the full scroll_pause every time
            try:
                WebDriverWait(driver, scroll_pause, poll_frequency=0.25).until(
                    lambda d: d.execute_script("return document.body.scrollHeight")
                    != last_height
                )
            except Exception:
                # height never changed within scroll_pause -> we're at the bottom
                break

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        cards = driver.find_elements("css selector", "div.thumbnail")

        seen = set()
        products = []

        for card in cards:
            try:
                title = card.find_element("css selector", "a.title").get_attribute(
                    "title"
                )
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
