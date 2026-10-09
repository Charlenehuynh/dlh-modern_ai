#!/usr/bin/env python3
"""Module that loads the Emotion dataset from the Hugging Face Hub."""
from datasets import load_dataset


def load_emotion_dataset():
    """
    Loads the dair-ai/emotion dataset.

    Returns:
        DatasetDict containing the train, validation, and test splits.
    """
    return load_dataset("dair-ai/emotion")