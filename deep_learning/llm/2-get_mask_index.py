#!/usr/bin/env python3
"""function that identifies and returns the positions of all <mask> tokens"""

import torch


def get_mask_index(inputs, tokenizer):
    """Return:
    mask_indices (list[int]): A list containing the index of every <mask> token
    """
    token_ids = inputs["input_ids"][0]
    mask_id = tokenizer.mask_token_id
    mask_indices = torch.where(token_ids == mask_id)[0].tolist()
    if not mask_indices:
        raise ValueError("No <mask> token found in the input!")
    return mask_indices
