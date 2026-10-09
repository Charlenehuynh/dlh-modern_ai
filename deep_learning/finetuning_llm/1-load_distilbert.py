#!/usr/bin/env python3
"""Module that loads a DistilBERT tokenizer and classification model."""
import transformers


def load_distilbert(model_name, num_classes, id2label, label2id):
    """
    Loads a tokenizer and a sequence classification model.

    Args:
        model_name (str): Name of the pre-trained DistilBERT to load.
        num_classes (int): Total number of output classes.
        id2label (dict): Maps numeric label IDs to label names.
        label2id (dict): Maps label names to numeric label IDs.

    Returns:
        tokenizer, model
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )
    return tokenizer, model
