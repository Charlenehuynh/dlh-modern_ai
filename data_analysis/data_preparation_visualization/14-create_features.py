#!/usr/bin/env python3
"""
This module contains the function create_features, which engineers new features
from the Telco Customer Churn dataset.
"""

import pandas as pd


def create_features(df):
    """
    Engineers new features from the dataset.
    """
    df_copy = df.copy()

    service_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df_copy["NumServices"] = 0

    for col in service_columns:
        if col == "InternetService":
            df_copy["NumServices"] += (
                df_copy[col].isin(["DSL", "Fiber optic"]).astype(int)
            )
        else:
            df_copy["NumServices"] += (df_copy[col] == "Yes").astype(int)

    # Change the lowest boundary to -1 to safely include 0 tenure values in the "0-12" group
    bins = [-1, 12, 24, 48, 60, float("inf")]
    labels = ["0-12", "13-24", "25-48", "49-60", "60+"]
    df_copy["TenureGroup"] = pd.cut(
        df_copy["tenure"], bins=bins, labels=labels, right=True
    )

    columns_to_drop = service_columns + ["tenure"]
    df_copy.drop(columns=columns_to_drop, inplace=True)

    return df_copy
