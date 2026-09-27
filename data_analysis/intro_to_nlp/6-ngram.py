#!/usr/bin/env python3
"""function that generates n-grams from a token list."""

import nltk


def generate_ngrams(tokens, n=2):
    """Arguments:

    tokens (list[str]): List of tokens used to generate n-grams.
    n (int): Size of each n-gram.
    """
    if not isinstance(tokens, list) or len(tokens) < n:
        return []
    ngram_list = list(nltk.ngrams(tokens, n))
    new_list = []
    for item in ngram_list:
        word = "_".join(item)
        new_list.append(word)
    return new_list
