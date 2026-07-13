#!/usr/bin/env python3
"""Function handles missing values in TotalCharges"""


def clean_total_charges(df, method="drop"):
    """Function handles missing values in TotalCharges"""
    df = df.copy()
    if method == "drop":
        df = df.dropna(subset=["TotalCharges"])
    elif method == "median":
        median_value = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_value)
    elif method == "impute":
        replace = df["MonthlyCharges"] * df["tenure"]
        df["TotalCharges"] = df["TotalCharges"].fillna(replace)
    return df
