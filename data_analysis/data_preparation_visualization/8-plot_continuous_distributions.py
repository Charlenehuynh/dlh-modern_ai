#!/usr/bin/env python3
"""function visualizes the distributions of continuous numerical features"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """Plot"""
    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3 * n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(columns_to_plot):
        data = df[col].dropna()
        # left subplot: Histogram + KDE
        ax_hist = axes[i, 0]
        ax_hist.hist(data, bins=30, density=True, alpha=0.7, edgecolor="black")

        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax_hist.plot(x_range, kde(x_range), color="red", linestyle="--")
        ax_hist.set_title(f"{col} Histogram + KDE")
        # Right subplot
        ax_box = axes[i, 1]
        ax_box.boxplot(data)
        ax_box.set_title(f"{col} Boxplot")

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
