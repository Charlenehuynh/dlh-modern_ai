#!/usr/bin/env python3
"""Decode the vocabulary tokens for each <mask> position."""


def decode_mask_predictions(mask_logits_list, tokenizer):
    """
    Maps every token ID in the vocabulary to its string, for each mask.

    """
    decoded_tokens = []
    for logits in mask_logits_list:
        vocab_size = logits.shape[-1]
        ids = [[i] for i in range(vocab_size)]
        tokens = [t.strip() for t in tokenizer.batch_decode(ids)]
        decoded_tokens.append(tokens)
    return decoded_tokens
