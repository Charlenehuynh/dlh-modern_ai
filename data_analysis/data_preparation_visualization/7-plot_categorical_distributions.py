#!/usr/bin/env python3
"""function visualizes categorical feature distributions"""

import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """plot distribution"""
    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(include="object").columns.tolist()
        if "Churn" in columns_to_plot:
            columns_to_plot.remove("Churn")
    else:
        columns_to_plot = columns_to_plot
    n_cols = 3
    n_rows = -(-len(columns_to_plot) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        counts = df[col].value_counts()
        axes[i].bar(counts.index, counts.values)
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
