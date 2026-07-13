#!/usr/bin/env python3
"""Function to remove duplicate"""


def remove_duplicates(df):
    """Function to remove duplicate"""
    df = df.drop_duplicates()
    return df
