#!/usr/bin/env python3
"""Function that removes stopwords a token list."""

import nltk


def remove_stopwords(tokens, language="english", extra_words=None, keep_words=None):
    """
    Arguments:
    tokens (list[str]): List of tokens to filter.
    language (str): NLTK stopword language to load. Defaults to "english".
    extra_words (set[str] | None): Additional words to add to the stopword set.
    keep_words (set[str] | None): Words to exclude from the stopword set

    The function should:
    Return [] if tokens is not a list.
    Load the NLTK stopword list for the given language.
    Add any words in extra_words to the stopword set.
    Remove any words in keep_words from the stopword set (e.g. it is used to preserve spam-indicative words that NLTK would otherwise silently discard).
    Return the filtered token list.
    Imports: import nltk
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
