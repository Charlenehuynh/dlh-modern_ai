#!/usr/bin/env python3
"""function that visualizes churn class distribution:"""

import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    df: pandas DataFrame with a Churn column
    Generates a bar plot of Churn value counts
    Uses colors: skyblue for ('No'), salmon for ('Yes')
    Displays the plot
    Returns: None
    """
    plt.figure(figsize=(12, 8))
    count = df["Churn"].value_counts()
    plt.title("Churn Distribution")
    colors = ["skyblue", "salmon"]
    plt.ylabel('Count')
    plt.bar(count.index, count.values, color=colors)
    plt.show()
