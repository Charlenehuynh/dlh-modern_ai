#!/usr/bin/env python3
"""Function that performs welch test"""

from scipy import stats


def ttest_numeric(df):
    """Returns a dictionary: {feature_name: p_value}"""
    numeric_cols = df.select_dtypes(include=["number"]).columns

    results = {}
    for col in numeric_cols:
        group_yes = df[df["Churn"] == "Yes"][col].dropna()
        group_no = df[df["Churn"] == "No"][col].dropna()
        _, p_val = stats.ttest_ind(group_yes, group_no, equal_var=False)
        results[col] = p_val

    return results
