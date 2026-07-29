#!/usr/bin/env python3
"""Module to create SHAP explainer and compute SHAP values"""

import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """Creates a SHAP explainer using X_train as background data
    and computes SHAP values for X_test."""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
