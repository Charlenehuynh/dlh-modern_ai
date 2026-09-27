#!/usr/bin/env python3
"""Word2Vec message embeddings"""
import numpy as np
import gensim.models


def word2vec_embeddings(corpus_tokens, vector_size=100, window=5,
                        min_count=2, sg=0, epochs=10, workers=4):
    """Trains Word2Vec and returns per-message embeddings."""
    model = gensim.models.Word2Vec(
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
        vectors = [model.wv[t] for t in tokens if t in model.wv]
        if vectors:
            X[i] = np.mean(vectors, axis=0)

    return X, model
