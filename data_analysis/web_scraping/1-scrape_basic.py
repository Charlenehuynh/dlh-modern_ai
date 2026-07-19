#!/usr/bin/env python3
"""scrapes the first page of quotes"""

from bs4 import BeautifulSoup

fetch_html = __import__("0-fetch_html").fetch_html


def scrape_basic(url):
    """
    url: quotes list endpoint
    return a list of dicts
    """
    fetch = fetch_html(url=url)
    soup = BeautifulSoup(fetch, "html.parser")
    items = soup.find_all("div", class_="quote")
    for quote in items:
        text = quote.find("span", class_="text")
        author = quote.find("small", class_="author")
        tags = quote.find("meta", class_="keywords")
        dict = {"text": text, "author": author, "tags": tags}
        return dict
