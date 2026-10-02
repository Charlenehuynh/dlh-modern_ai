#!/usr/bin/env python3
"""Find the positions of <mask> tokens in tokenized input"""


def get_mask_index(inputs, tokenizer):
    """Return a list with the index of every <mask> token."""
    token_ids = inputs["input_ids"][0].tolist()
    mask_id = tokenizer.mask_token_id
    mask_indices = [i for i, t in enumerate(token_ids) if t == mask_id]
    if not mask_indices:
        raise ValueError("No <mask> token found in the input!")
    return mask_indices