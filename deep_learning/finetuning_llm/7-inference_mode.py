#!/usr/bin/env python3
"""Module that creates a text classification inference pipeline."""
import transformers


def inference_mode(model_path, top_k):
    """
    Initializes a text classification pipeline from a fine-tuned model.

    Args:
        model_path (str): Path to the saved model and tokenizer.
        top_k (int): Number of top predictions to return per input.

    Returns:
        transformers.Pipeline ready to classify new texts.
    """
    return transformers.pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        top_k=top_k
    )
