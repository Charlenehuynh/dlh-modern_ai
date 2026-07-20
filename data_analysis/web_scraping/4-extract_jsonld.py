#!/usr/bin/env python3
"""function that pulls quotes from embedded JSON‑LD on a page"""

import json
from bs4 import BeautifulSoup

fetch_html = __import__('0-fetch_html').fetch_html


def extract_jsonld(url):
    """
    return a list of quote dicts
    """
    fetch = fetch_html(url)
    soup = BeautifulSoup(fetch, "html.parser")
    items = soup.find_all("script", type="application/ld+json")
    temp = []
    for i in items:
        dict1 = json.loads(i.text)
        temp.append({'text': dict1.get("text"),
                     'author': dict1.get("author", {}).get("name"),
                     'tags': dict1.get("keywords")})
    return temp
