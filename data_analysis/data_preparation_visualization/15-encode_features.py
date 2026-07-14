#!/usr/bin/env python3
"""
This module contains the function create_features.
It engineers new features using the Telco Customer Churn dataset.
"""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Engineers new features from the dataset.
    """
    df_enc = df.copy()

    churn_le = preprocessing.LabelEncoder()
    df_enc["Churn"] = churn_le.fit_transform(df_enc["Churn"])

    binary_cols = ["Partner", "Dependents", "PaperlessBilling", "SeniorCitizen"]

    df_enc["SeniorCitizen"] = df_enc["SeniorCitizen"].map(
        {0: "No", 1: "Yes", "No": "No", "Yes": "Yes"}
    )

    binary_oe = preprocessing.OrdinalEncoder(categories=[["No", "Yes"]], dtype=int)

    for col in binary_cols:
        df_enc[col] = binary_oe.fit_transform(df_enc[[col]])

    sorted_tenure_categories = sorted(df["TenureGroup"].unique().tolist())
    tenure_oe = preprocessing.OrdinalEncoder(
        categories=[sorted_tenure_categories], dtype=int
    )
    df_enc["TenureGroup"] = tenure_oe.fit_transform(df_enc[["TenureGroup"]])

    oh_cols = ["Contract", "PaymentMethod"]
    df_enc = pd.get_dummies(df_enc, columns=oh_cols, drop_first=True, dtype=int)

    return df_enc, churn_le, binary_oe, tenure_oe
