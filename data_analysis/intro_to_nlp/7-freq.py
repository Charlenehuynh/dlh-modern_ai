#!/usr/bin/env python3
"""function that plots the most frequent tokens in a preprocessed corpus."""

import nltk
import matplotlib.pyplot as plt


def plot_top_n_frequencies(corpus_tokens, n=20):
    """return the full frequency distribution object"""
    # Flatten the list of lists into a single token list and computes freq.
    new_list = []
    for i in corpus_tokens:
        new_list.extend(i)
    freq_obj = nltk.FreqDist(new_list)
    words_freq = freq_obj.most_common(n)
    word_list = []
    freq_list = []
    for words, frequency in words_freq:
        word_list.append(words)
        freq_list.append(frequency)
    # Plot a bar chart using plt.bar
    plt.figure(figsize=(12, 5))
    plt.bar(word_list, freq_list)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {n} Most Frequent Words")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()
    return freq_obj
