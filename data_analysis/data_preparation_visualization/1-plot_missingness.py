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
