#!/usr/bin/env python3
"""Bag-of-Words feature extraction"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2),
                 min_df=2, max_df=0.95, binary=False):
    """Builds a Bag-of-Words feature matrix from a list of token lists."""
    docs = [" ".join(tokens) for tokens in corpus_tokens]

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
    )
    X = vectorizer.fit_transform(docs)
    return X, vectorizer
