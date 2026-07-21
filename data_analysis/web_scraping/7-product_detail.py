#!/usr/bin/env python3
"""Scrape a single product detail page from webscraper.io."""

import time
from selenium import webdriver


def scrape_product_detail(url, delay=2.0):
    """Open a product detail page, wait, and return its details."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(delay)

        # Title: second <h4> inside .caption
        h4_elements = driver.find_elements("css selector", ".caption h4")
        title = h4_elements[1].text

        # Price: first <h4 class="price">
        price = driver.find_element("css selector", "h4.price").text

        # Description
        description = driver.find_element("css selector", "p.description").text

        # Rating: count of star icons inside .ratings
        stars = driver.find_elements("css selector",
                                     ".ratings p.ws-icon.ws-icon-star")
        rating = len(stars)

        return {
            "title": title,
            "price": price,
            "description": description,
            "rating": rating,
        }
    finally:
        driver.quit()
