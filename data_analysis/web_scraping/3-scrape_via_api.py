#!/usr/bin/env python3
"""function fetches quote data from API pages"""

import json

fetch_html = __import__('0-fetch_html').fetch_html


def scrape_via_api(base_url):
    """return a list of quote dicts"""
    all_quote = []
    starting_page = 1
    while True:
        current_url = base_url + "/api/quotes?page=" + str(starting_page)
        current_html = fetch_html(current_url)
        dict1 = json.loads(current_html)
        quote_list = dict1["quotes"]
        for i in quote_list:
            temp = {"text": i["text"], "author": i["author"]["name"], "tags": i["tags"]}
            all_quote.append(temp)
        starting_page += 1
        if dict1["has_next"] is False:
            break
    return all_quote
