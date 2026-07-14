#!/usr/bin/env python3
"""function that performs chi-square tests for categorical features"""

import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Returns a dictionary: {feature_name: p_value}"""
    df_temp = df.copy()

    if "SeniorCitizen" in df_temp.columns:
        df_temp["SeniorCitizen"] = df_temp["SeniorCitizen"].astype(str)

    df_sel = df_temp.select_dtypes(include=["object"])
    df_sel = df_sel.drop(columns=["Churn"], errors="ignore")

    chi_square = {}

    for col in df_sel.columns:
        contingency = pd.crosstab(df_temp[col], df_temp["Churn"])
        _, p, _, _ = stats.chi2_contingency(contingency)
        chi_square[col] = p

    return chi_square
