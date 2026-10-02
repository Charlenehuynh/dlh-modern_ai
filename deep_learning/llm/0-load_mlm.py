#!/usr/bin/env python3
"""that loads a pre-trained RoBERTa model"""

import transformers


def load_mlm(model_name):
    """return an instance of RobertaForMaskedLM ready for inference."""
    model = transformers.RobertaForMaskedLM.from_pretrained(model_name)
    model.eval()
    return model
