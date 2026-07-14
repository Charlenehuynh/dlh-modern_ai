#!/usr/bin/env python3
"""
function to scale numeric features in the Telco Customer Churn dataset
for machine learning modeling.
"""

from sklearn import preprocessing


def scale_numeric(df):
    """
    Standardizes numeric features using Scikit-learn's StandardScaler.
    """
    df_scaled = df.copy()

    num_cols = ["MonthlyCharges", "TotalCharges"]
    scaler = preprocessing.StandardScaler()
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

    return df_scaled
