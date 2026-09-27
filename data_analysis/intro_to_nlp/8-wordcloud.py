#!/usr/bin/env python3
"""function that generates a word cloud from a preprocessed corpus."""

import wordcloud
import matplotlib.pyplot as plt


def generate_wordcloud(corpus_tokens, max_words=200, label=None):
    """Return: the fitted wordcloud object."""
    all_tokens = []
    for i in corpus_tokens:
        all_tokens.extend(i)
    new_list = " ".join(all_tokens)
    wc = wordcloud.WordCloud(
        max_words=max_words, width=800, height=400, random_state=42
    )
    wc.generate(new_list)

    return wc
