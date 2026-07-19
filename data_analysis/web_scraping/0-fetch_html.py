#!/usr/bin/env python3

"""function that fetches a web page and returns its HTML as text"""

import requests


def fetch_html(url, headers=None, timeout=10):
    """_summary_

    Args:
        url (text): page to retrieve
        headers (text):dict of HTTP headers e.g. {"User-Agent": "…”}
        timeout (int):number of s to wait before aborting. Defaults:10.
    """
    response = requests.get(url=url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text
