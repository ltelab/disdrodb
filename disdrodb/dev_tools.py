#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:56:38 2022

@author: ghiggi
"""
import numpy as np 
import pandas as pd 
from disdrodb.data_encodings import get_L0_dtype_standards 

def print_df_first_n_rows(df, n=5, column_names=True):
    columns = list(df.columns) 
    for i in range(len(df.columns)): 
        if column_names:
            print(" - Column", i, "(", columns[i], "):")
        else:
            print(" - Column", i, ":")    
        print("     ", df.iloc[0:(n+1), i].values)
    return None

def print_df_random_n_rows(df, n=5, column_names=True):
    columns = list(df.columns)
    df = df.copy() 
    df = df.sample(n=n)
    for i in range(len(df.columns)): 
        if column_names:
            print(" - Column", i, "(", columns[i], "):")
        else: 
            print(" - Column", i, ":")
        print("     ", df.iloc[0:(n+1), i].values)
    return None

def print_df_column_names(df):
    for i, column in enumerate(df.columns):
        print(" - Column",i,":", column)
    return None

def print_valid_L0_column_names():
    print(list(get_L0_dtype_standards().keys()))
    return None

def _check_valid_column_index(column_idx, n_columns):
    if column_idx > (n_columns-1):
        raise ValueError("'column_idx' must be between 0 and {}".format(n_columns-1))
    if column_idx < 0: 
        raise ValueError("'column_idx' must be between 0 and {}".format(n_columns-1))

def _check_columns_indices(column_indices, n_columns): 
    if not isinstance(column_indices, (int, list, type(None), slice)):
        raise TypeError("'column_indices' must be an integer, a list of integers, or None.")
    if column_indices is None: 
       column_indices = list(range(0, n_columns))
    if isinstance(column_indices, slice): 
        start = column_indices.start
        stop = column_indices.stop 
        step = column_indices.step
        step = 1 if step is None else step
        column_indices = list(range(start, stop, step))
    if isinstance(column_indices, list): 
        [_check_valid_column_index(idx, n_columns) for idx in column_indices]
    if isinstance(column_indices, int): 
        _check_valid_column_index(column_indices, n_columns)
        column_indices = [column_indices]
    return column_indices

def print_df_columns_unique_values(df, column_indices=None, column_names=True):
    # Retrieve column names 
    columns = list(df.columns) 
    n_columns = len(columns)
    # Checks for printing specific columns only
    column_indices = _check_columns_indices(column_indices, n_columns)
    columns = [columns[idx] for idx in column_indices]
    # Printing         
    for i, column in zip(column_indices, columns): 
        if column_names:
            print(" - Column", i, "(", column, "):")
        else:
            print(" - Column", i, ":")
        print("     ", df[column].unique().tolist())
    return None        

def get_df_columns_unique_values_dict(df, column_indices=None, column_names=True):
    """Create a dictionary {column: unique values} """
    # Retrieve column names 
    columns = list(df.columns) 
    n_columns = len(columns)
    # Checks for printing specific columns only
    column_indices = _check_columns_indices(column_indices, n_columns)
    columns = [columns[idx] for idx in column_indices]
    # Create dictionary       
    d = {}
    for i, column in zip(column_indices, columns): 
        if column_names:
            key = column
        else:
            key = "Column " + str(i)         
        d[key] = df[column].unique().tolist()
    # Return 
    return d 


def print_df_summary_stats(df, column_indices=None, column_names=True):
    # Define columns of interest
    columns = df.columns
    n_columns = len(columns)
    column_indices = _check_columns_indices(column_indices, n_columns)
    columns_of_interest = [columns[idx] for idx in column_indices]
    # Remove columns of dtype object or string 
    indices_to_remove = np.where((df.dtypes == type(object)) | (df.dtypes == str))
    indices = np.arange(0, len(df.columns))
    indices = indices[np.isin(indices,indices_to_remove, invert=True)]
    columns = df.columns[indices]
    if len(columns) == 0: 
        raise ValueError("No numeric columns in the dataframe.")
    # Select only columns of interest 
    idx_of_interest = np.where(np.isin(columns, columns_of_interest))
    if len(idx_of_interest) == 0: 
        raise ValueError("No numeric columns at the specified column_indices.")
    columns = columns[idx_of_interest]
    indices = indices[idx_of_interest]
    # Compute summary stats
    summary_stats = ["mean", "min", "25%", "50%", "75%", "max"]
    df_summary = df.describe()
    df_summary = df_summary.loc[summary_stats]
    # Print summary stats 
    for i, column in zip(indices, columns): 
        # Printing         
       
        if column_names:
            print(" - Column", i, "(", column, "):")
        else:
            print(" - Column", i, ":")
        tmp_df = df_summary[[column]]
        tmp_df.columns = ['']
        print("     ", tmp_df)
    return None       
 

def print_df_with_any_nan_rows(df):
    df_bool_is_nan = df.isnull()
    idx_nan_rows = df_bool_is_nan.any(axis=1)
    df_nan_rows = df.loc[idx_nan_rows]
    print_df_first_n_rows(df_nan_rows, n=len(df_nan_rows))
 
        
 