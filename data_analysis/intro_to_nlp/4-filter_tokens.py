#!/usr/bin/env python3
"""function that removes low-information tokens."""

import re

PLACEHOLDERRE = re.compile(r"^<[A-Za-z]+>$")


def filter_tokens(tokens, min_len=2, strip_hashtag=False):
    """
    Return list of qualified tokens
    """
    # 1.Return [] for an empty or falsy tokens input.
    if not tokens:
        return []

    new_list = []
    for t in tokens:
        # 2.Keep any token that matches _PLACEHOLDER_RE
        if PLACEHOLDERRE.match(t):
            new_list.append(t)
            continue
        # 3.If strip_hashtag=True, strip prefer #
        if strip_hashtag is True:
            if t.startswith("#"):
                t = t[1:]

        # 4.Drop tokens shorter than min len
        if len(t) < min_len:
            continue
        # drop tokens that contain no alphabetic character
        if not any(c.isalpha() for c in t):
            continue
        new_list.append(t)
    return new_list
