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
    plt.hist(df[df["Churn"] == "No"][col], bins=30, alpha=0.5, label="No")
    plt.hist(df[df["Churn"] == "Yes"][col], bins=30, alpha=0.5, label="Yes")
    plt.xlabel(col)
    plt.ylabel("Frequency")  # Standard for histograms
    plt.title(f"{col} Distribution by Churn")
    plt.legend(title="Churn")
    plt.show()
