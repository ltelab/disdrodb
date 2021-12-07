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

