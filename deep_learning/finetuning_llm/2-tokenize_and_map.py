#!/usr/bin/env python3
"""Module that tokenizes the Emotion dataset splits."""


def tokenize_and_map(dataset, tokenizer, max_length, truncation, batched):
    """
    Tokenizes the text of every split in the dataset.

    Args:
        dataset (DatasetDict): Dataset with 'train', 'validation', 'test'.
        tokenizer: Pretrained tokenizer.
        max_length (int): Maximum token length for truncation.
        truncation (bool): Whether to truncate longer sequences.
        batched (bool): Whether to process the dataset in batches.

    Returns:
        tokenized_train, tokenized_val, tokenized_test
    """
    def tokenize(examples):
        """Tokenizes a batch (or single example) of text."""
        return tokenizer(examples["text"],
                         truncation=truncation,
                         max_length=max_length)

    tokenized = dataset.map(tokenize, batched=batched)
    return tokenized["train"], tokenized["validation"], tokenized["test"]
