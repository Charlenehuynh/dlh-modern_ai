#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd

""" function that visualizes churn rates per category:"""


def plot_categorical_vs_churn(df, col):
    """
    df: pandas DataFrame with Churn column
    col: Categorical column name
    Uses a figure size of (12, 8)
    Adds a title to the plot in the format: "Churn Rate by <col>"
    Plots churn rate (Yes proportion) per category as a bar plot
    Sets y-axis label to "Churn Rate"
    Rotates the x-axis labels by 45°
    Displays the plot
    Returns: None
    """
    churn_rate = (df["Churn"] == "Yes").groupby(df[col]).mean().reset_index(name="Rate")
    labels = churn_rate[col]
    values = churn_rate["Rate"]
    plt.figure(figsize=(12, 8))
    plt.bar(labels, values)
    plt.ylabel("Churn Rate")
    plt.title(f"Churn Rate by {col}")
    plt.xticks(rotation=45)
    plt.show()

    return None
