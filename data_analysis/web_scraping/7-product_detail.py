import time
from selenium import webdriver


def scrape_product_detail(url, delay=2.0):
    browser_options = webdriver.ChromeOptions()
    browser_options.add_argument("--headless")
    browser_options.add_argument("--window-size=1920,1080")
    browser_options.add_argument("--no-sandbox")
    browser_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=browser_options)
    driver.get(url)
    time.sleep(delay)

    product = {}
    try:
        title = driver.find_element(
            "css selector", ".caption h4:nth-of-type(2)"
        ).text.strip()

        price = driver.find_element("class name", "price").text

        description = driver.find_element(
            "class name", "description"
        ).text.strip()

        rating_element = driver.find_elements(
            "css selector", ".ratings .ws-icon-star"
        )
        rating = len(rating_element)

        product = {
            "title": title,
            "price": price,
            "description": description,
            "rating": rating,
        }
    finally:
        driver.quit()

    return product