#!/usr/bin/env python3

"""function removes the customerID column"""


def drop_customerID(df):
    """
    df: pandas DataFrame containing a customerID column
    Drops the customerID column
    Returns the modified DataFrame
    """
    df = df.drop(columns="customerID")
    return df
