#!/usr/bin/env python3
"""function calculate importances from a trained random forest model."""

import numpy as np


def feature_importance(rf):
    """Get feature importances and their sorted indices"""
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
