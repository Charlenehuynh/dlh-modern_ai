#!/usr/bin/env python3
"""function that performs initial dataset exploration:"""

import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """Left subplot: bar chart of ham vs spam counts using sns.barplot:
    Returns: None
    Imports: import matplotlib.pyplot as plt and import seaborn as sns"""
    lengths = df["message"].str.len()
    counts = df["label"].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Left: ham vs spam counts
    sns.barplot(x=counts.index, y=counts.values, ax=ax1)
    ax1.set_title("Ham vs Spam Counts")
    ax1.set_xlabel("label")
    ax1.set_ylabel("count")

    # Right: histogram of raw message lengths
    sns.histplot(lengths, bins=50, ax=ax2)
    ax2.set_title("Histogram of Raw Message Lengths")
    ax2.set_xlabel("length")
    ax2.set_ylabel("count")

    plt.tight_layout()
    plt.show()
