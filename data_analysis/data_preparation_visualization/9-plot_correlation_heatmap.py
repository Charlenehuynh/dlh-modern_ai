#!/usr/bin/env python3
"""plot seaborn"""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    df: pandas DataFrame
    Computes pairwise correlations
    Generates an annotated heatmap with coolwarm colormap
    Set vmin = -1 and vmax = 1
    Displays the plot
    Returns: None
    """
    plt.figure(figsize=(6, 5))
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt="0.2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
