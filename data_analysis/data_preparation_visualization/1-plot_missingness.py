#!/usr/bin/env python3
"""Visualizes missing values"""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualizes missing values"""
    plt.figure(figsize=(12, 8))

    for position, item in enumerate(df.columns):
        missing_rows = df[item][df[item].isnull()].index
        plt.scatter(x=missing_rows, y=[position] * len(missing_rows), marker="|")
    plt.yticks(range(len(df.columns)), df.columns)

    plt.tight_layout()
    plt.show()


## TEST
import pandas as pd

data = {
    "customerID": ["A", "B", "C", "A", "D", "E"],
    "gender": ["Male", "Female", np.nan, "Male", "Female", "Male"],
    "Partner": ["Yes", "No", "No", np.nan, "Yes", None],
    "tenure": [1, 24, 5, 1, 36, np.nan],
    "TotalCharges": [29.85, 2138.40, np.nan, 29.85, 3803.40, 0],
    "Churn": ["No", "Yes", np.nan, "No", None, "No"],
}
test_df = pd.DataFrame(data)
# print(test_df)
plot_missingness(test_df)
