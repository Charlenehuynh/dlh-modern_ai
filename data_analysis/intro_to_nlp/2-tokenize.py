#!/usr/bin/env python3
"""Function that tokenizes a cleaned SMS message."""

import nltk

EMOTICON_MAP = {
    "<3": "<EMO>",
    "</3": "<EMO>",
    ":)": "<EMO>",
    ":-)": "<EMO>",
    ":(": "<EMO>",
    ":-(": "<EMO>",
    ":d": "<EMO>",
    ";)": "<EMO>",
    ":|": "<EMO>",
    ">:(": "<EMO>",
    ":p": "<EMO>",
    "b)": "<EMO>",
    "o:)": "<EMO>",
}


def normalize_emoticons(tokens, emoticon_action="replace"):
    if not isinstance(tokens, list):
        return []

    result = []

    for token in tokens:
        mapped = EMOTICON_MAP.get(token.lower())

        if mapped:
            if emoticon_action == "replace":
                result.append(mapped)
        else:
            result.append(token)

    return result


def tokenize_text(text, method="tweet"):
    """that tokenizes a cleaned SMS message."""
    # Return an empty list if text is not a string
    if not isinstance(text, str):
        return []
    if method == "tweet":
        tk = nltk.tokenize.TweetTokenizer(reduce_len=True)
        tk = tk.tokenize(text)
    elif method == "word":
        tk = nltk.word_tokenize(text)
    elif method == "split":
        tk = text.split()
    else:
        raise ValueError("Invalid tokenizer method")
    return tk
