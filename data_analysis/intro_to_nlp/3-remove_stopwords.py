#!/usr/bin/env python3
"""Function that removes stopwords a token list."""

import nltk


def remove_stopwords(tokens, language="english", extra_words=None, keep_words=None):
    """
    Remove stop words
    """
    if not isinstance(tokens, list):
        return []
    words = nltk.corpus.stopwords.words(language)
    set_words = set(words)
    if extra_words is not None:
        set_words.update(extra_words)
    if keep_words is not None:
        set_words.difference_update(keep_words)
    allowed = [i for i in tokens if i not in set_words]
    return allowed
