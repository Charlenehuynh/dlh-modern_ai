#!/usr/bin/env python3
"""computes the raw logits for all <mask>"""

import torch


def compute_mask_logits(model, inputs, mask_indices):
    """Return:
    mask_logits_list (list[torch.Tensor])
    mask token
    """
    model.eval()
    with torch.no_grad():
        outputs = model(**input)
    logits = outputs.logits[0]
    return [logits[idx] for idx in mask_indices]
