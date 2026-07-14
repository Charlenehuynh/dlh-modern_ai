#!/usr/bin/env python3
"""
function encode features in the Telco Customer Churn dataset for machine learning modeling.
"""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Encodes features for modeling using Scikit-learn preprocessing tools.
    """
    df_enc = df.copy()

    churn_le = preprocessing.LabelEncoder()
    df_enc["Churn"] = churn_le.fit_transform(df_enc["Churn"])

    binary_cols = ["Partner", "Dependents", "PaperlessBilling", "SeniorCitizen"]
    binary_oe = preprocessing.OrdinalEncoder(
        categories=[["No", "Yes"]] * len(binary_cols)
    )
    df_enc[binary_cols] = binary_oe.fit_transform(df_enc[binary_cols]).astype(int)

    sorted_tenure_categories = sorted(df["TenureGroup"].unique().tolist())
    tenure_oe = preprocessing.OrdinalEncoder(categories=[sorted_tenure_categories])

    df_enc["TenureGroup"] = tenure_oe.fit_transform(df_enc[["TenureGroup"]])

    oh_cols = ["Contract", "PaymentMethod"]
    df_enc = pd.get_dummies(df_enc, columns=oh_cols, drop_first=True, dtype=int)

    return df_enc, churn_le, binary_oe, tenure_oe
