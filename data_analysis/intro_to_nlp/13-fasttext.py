#!/usr/bin/env python3
"""FastText message embeddings"""
import numpy as np
import gensim.models


def fasttext_embeddings(corpus_tokens, vector_size=100, window=5,
                        min_count=1, sg=0, epochs=10, workers=4):
    """Trains FastText and returns per-message embeddings."""
    model = gensim.models.FastText(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,
    )

    X = np.zeros((len(corpus_tokens), vector_size))
    for i, tokens in enumerate(corpus_tokens):
        if tokens:
            X[i] = np.mean([model.wv[t] for t in tokens], axis=0)

    return X, model
