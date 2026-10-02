#!/usr/bin/env python3

"""creates a high-level interface for performing language translation"""

import transformers


def translate_text(model_name, src_lang=None, tgt_lang=None):
    """return translator: A Hugging Face pipeline object."""
    task = "translation"
    if src_lang and tgt_lang:
        task = f"translation_{src_lang}_to_{tgt_lang}"
    translator = transformers.pipeline(task, model=model_name)
    return translator
