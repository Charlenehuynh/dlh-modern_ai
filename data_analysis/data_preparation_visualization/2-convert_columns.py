#!/usr/bin/env python3
"""function performs type conversion for specific column"""

import pandas as pd


def convert_columns(df):
    """function performs type conversion for specific column"""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df
