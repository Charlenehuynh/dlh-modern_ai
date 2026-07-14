#!/usr/bin/env python3

import pandas as pd

""" function contains function for new features from the Telco Customer Churn dataset."""


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

    bins = [0, 12, 24, 48, 60, float("inf")]
    labels = ["0-12", "13-24", "25-48", "49-60", "60+"]
    df_copy["TenureGroup"] = pd.cut(
        df_copy["tenure"], bins=bins, labels=labels, right=True
    )

    columns_to_drop = service_columns + ["tenure"]
    df_copy.drop(columns=columns_to_drop, inplace=True)

    return df_copy
