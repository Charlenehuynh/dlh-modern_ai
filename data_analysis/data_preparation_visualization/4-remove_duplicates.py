#!/usr/bin/env python3

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df