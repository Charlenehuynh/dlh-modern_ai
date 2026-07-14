#!/usr/bin/env python3
"""function compares continuous numeric feature distributions by churn"""

import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
    df: pandas DataFrame with Churn column
    col: Numeric column name
    Uses a figure size of (12, 8)
    Adds a title to the plot in the format: "<col> Distribution by Churn"
    Plots overlapping histograms for Churn=Yes and Churn=No
    Sets the x-axis label to "<col>"
    Uses 30 bins to group the data along the x-axis
    Adds a legend with a title
    Displays the plot
    Returns: None
    """
    plt.figure(figsize=(12, 8))
    col_no = df[df["Churn"] == "No"][col]
    col_yes = df[df["Churn"] == "Yes"][col]
    plt.figure(figsize=(12, 8))
    plt.hist([col_no, col_yes], bins=30, label=["No", "Yes"])
    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn")
    plt.show()
