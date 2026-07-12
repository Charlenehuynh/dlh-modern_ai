#!/usr/bin/env python3
"""Visualizes missing values"""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """ Visualizes missing values"""
    plt.figure(figsize=(12, 8))

    plt.figure(figsize=(12, 8))

    df_nul = df.isnull()
    nul = np.where(df_nul)

    x = nul[0]
    y = nul[1]

    plt.scatter(x, y, marker="|")

    plt.title("Missingness Plot")
    plt.yticks(np.arange(0, len(df.columns.values)), df.columns.values)

    plt.tight_layout()
    plt.show()
    return None
