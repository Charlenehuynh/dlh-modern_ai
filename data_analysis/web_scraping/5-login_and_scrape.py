#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup

""" function that logs in and scrapes quotes after authentication"""


def login_and_scrape(login_url, user, pwd):
    """Return: a list of quote dicts"""
    session = requests.Session()
    login_page = session.get(login_url).text
    soup = BeautifulSoup(login_page, "html.parser")
    token_tag = soup.find("input", attrs={"name": "csrf_token"})
    token_str = token_tag["value"]
    dict1 = {"username": user, "password": pwd, "csrf_token": token_str}
    login_response = session.post(url=login_url, data=dict1)
    quote_page = session.get("https://quotes.toscrape.com/").text
    soup_quote = BeautifulSoup(quote_page, "html.parser")
    items = soup_quote.find_all("div", class_="quote")
    results = []
    for quote in items:
        text = quote.find("span", class_="text").text
        author = quote.find("small", class_="author").text
        tags = quote.find_all("a", class_="tag")
        list_tag = [i.text for i in tags]
        dict1 = {"text": text, "author": author, "tags": list_tag}
        results.append(dict1)
    return results
