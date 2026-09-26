#!/usr/bin/env python3
"""function normalises tokens via lemmatisation or stemming"""

import nltk
import re


def get_pos(tag):
    """get the pos of tag"""
    if tag.startswith("V"):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith("J"):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith("R"):
        return nltk.corpus.wordnet.ADV
    return nltk.corpus.wordnet.NOUN


def normalize_tokens(tokens, method="lemmatize"):
    """
    Arguments:
    tokens (list[str]): List of tokens to normalize.
    method (str): Normalization method. Must be "lemmatize" or "stem".
    """
    if method != "lemmatize" and method != "stem":
        raise ValueError("method must be 'lemmatize' or 'stem'")
    pattern = r"<[A-Z]+>"
    new_list = []

    ps = nltk.PorterStemmer()
    lem = nltk.WordNetLemmatizer()
    if method == "stem":
        for t in tokens:
            if re.fullmatch(pattern, t):
                new_list.append(t)
            else:
                new_list.append(ps.stem(t))
    elif method == "lemmatize":
        tag_tuble = nltk.pos_tag(tokens)
        for word, tag in tag_tuble:
            if re.fullmatch(pattern, word):
                new_list.append(word)
            else:
                tagged = get_pos(tag)
                new_list.append(lem.lemmatize(word, pos=tagged))
    return new_list
