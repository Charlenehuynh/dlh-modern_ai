#!/usr/bin/env python3
"""Module that creates a dynamic padding data collator."""
import transformers


def create_data_collator(tokenizer):
    """
    Creates a data collator that pads each batch to its longest sequence.

    Args:
        tokenizer: Pretrained tokenizer (e.g., DistilBERT tokenizer).

    Returns:
        DataCollatorWithPadding instance.
    """
    return transformers.DataCollatorWithPadding(tokenizer=tokenizer)
