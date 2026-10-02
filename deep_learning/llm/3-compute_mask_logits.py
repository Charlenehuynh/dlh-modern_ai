#!/usr/bin/env python3
"""computes the raw logits for all <mask>"""

import torch


def compute_mask_logits(model, inputs, mask_indices):
    """Return:
    mask_logits_list (list[torch.Tensor])
    mask token
    """
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs[0][0]
    return [logits[int(idx)] for idx in mask_indices]
