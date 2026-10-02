#!/usr/bin/env python3
"""function that loads a pre-trained RoBERTa tokenizer."""

import transformers


def tokenize_text(model_name, sentence, padding=True):
    """
    Return:
    tokenizer: An instance of RobertaTokenizer.
    inputs: Tokenized representation of the sentence as PyTorch tensors.
    """
    tok = transformers.RobertaTokenizer.from_pretrained(model_name)
    result = tok(sentence, padding=padding, return_tensors="pt")
    return tok, result
