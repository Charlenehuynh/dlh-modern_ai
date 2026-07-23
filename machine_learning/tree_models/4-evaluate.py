#!/usr/bin/env python3
"""Generate a classification report for a trained classifier"""

from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """Generate a detailed classification report"""
    report = metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=list(class_names),
    )
    return report
