#!/usr/bin/env python3
"""follows “Next” links on web until no more pages remain"""

from bs4 import BeautifulSoup
import time
from urllib import parse

fetch_html = __import__("0-fetch_html").fetch_html
scrape_basic = __import__("1-scrape_basic").scrape_basic


def scrape_paginated(base_url):
    """return the full list of quote dicts"""
    """
    start with first url
    on each page
    1. get the quote from that page (task 1)
    2. Check there is a next link -> continue
    3. next link only give/page/2/ -> turn into full URL (using urllib.parse)
    4. loop until no next ink
    5. When move to a new page -> short pause (time.sleep)
    6. create a single list contain every quote from every page combined
    """
    scrape = fetch_html(base_url)
    soup = BeautifulSoup(scrape, "html.parser")
    result = scrape_basic(base_url)
    list1 = []
    list1.extend(result)
    while soup.find("li", class_="next"):
        href = soup.find("li", class_="next").find("a").get("href")
        new_url = parse.urljoin(base_url, href)
        new_html = fetch_html(new_url)
        soup = BeautifulSoup(new_html, "html.parser")
        result = scrape_basic(new_url)
        time.sleep(5)
        list1.extend(result)
    return list1
