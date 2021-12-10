#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:16:59 2021

@author: kimbo
"""


def print_unique(df):
    '''
    Return all unique the unique values of a dataframe into a dictionary
    '''
    a = {}

    for col in list(df):
        a[col] = df[col].unique()
        
    for key, value in a.items():
        print(key, ' : ', value)
        
def print_nan_rows(df):
    is_NaN = df.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = df[row_has_NaN]

    print(rows_with_NaN)
    

