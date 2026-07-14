#!/usr/bin/env python3
"""function that performs chi-square tests for categorical features"""

import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Returns a dictionary: {feature_name: p_value}"""
    p_values = {}

    for col in df.columns:
        if col == "Churn":
            continue
        contingency_table = pd.crosstab(df[col], df["Churn"])
        _, p_val, _, _ = stats.chi2_contingency(contingency_table)
        p_values[col] = p_val

    return p_values
