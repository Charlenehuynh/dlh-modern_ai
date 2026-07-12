#!/usr/bin/env python3
"""Visualizes missing values"""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    plt.figure(figsize=(12, 8))

    for col_idx, col in enumerate(df.columns):
        missing_rows = np.where(df[col].isnull())[0]
        plt.scatter(
            missing_rows,
            np.full(len(missing_rows), col_idx),
            marker="|",
            s=100,
            color="black",
        )

    plt.title("Missingness Plot")
    plt.xlabel("Row Index")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.tight_layout()
    plt.show()

    return None
