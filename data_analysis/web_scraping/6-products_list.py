#!/usr/bin/env python3
"""that opens a product page and returns a list of product dictionaries"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By


def scrape_products(url):
    """Do not use BeautifulSoup or regex"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    products = []
    try:
        driver.get(url)

        containers = driver.find_elements(By.CSS_SELECTOR, ".thumbnail")
        for container in containers:
            title_elment = container.find_element(By.CSS_SELECTOR, "a.title")
            title = title_elment.get_attribute("title")
            price = container.find_element(By.CSS_SELECTOR, "h4.price").text
            description = (container.find_element
                        (By.CSS_SELECTOR, "p.description").text)
            rating_element = container.find_element(
                By.CSS_SELECTOR, ".ratings p[data-rating]"
            )
            rating = int(rating_element.get_attribute("data-rating"))
            products.append(
                {
                    "title": title,
                    "price": price,
                    "description": description,
                    "rating": rating,
                }
            )
    finally:
        driver.quit()
    return products
