#!/usr/bin/env python3
"""Module that computes classification evaluation metrics."""

import numpy as np
import sklearn.metrics


def compute_metrics(predictions):
    """
    Computes accuracy, precision, recall and F1-score (weighted average).

    Args:
        predictions: A transformers.EvalPrediction object with
            `predictions` (logits) and `label_ids` (true labels).

    Returns:
        dict with keys 'accuracy', 'precision', 'recall', 'f1'.
    """
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=-1)

    accuracy = sklearn.metrics.accuracy_score(labels, preds)
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
