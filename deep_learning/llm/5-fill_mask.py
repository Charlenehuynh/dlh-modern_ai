#!/usr/bin/env python3


import transformers

"""creates a high-level interface for performing MLL """


def fill_mask(model_name, top_k):
    """return fill: A Hugging Face pipeline object."""
    analyzer = transformers.pipeline("fill-mask", top_k=top_k, model=model_name)
    return analyzer
