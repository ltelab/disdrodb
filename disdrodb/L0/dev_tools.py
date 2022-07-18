#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:56:38 2022

@author: ghiggi
"""
import numpy as np
import pandas as pd
from disdrodb.L0.standards import get_L0A_dtype
from disdrodb.L0.check_standards import (
    get_field_nchar_dict,
    get_field_ndigits_dict,
    get_field_ndigits_decimals_dict,
    get_field_ndigits_natural_dict,
)

def check_column_names(column_names, sensor_name): 
    "Checks that the columnn names respects DISDRODB standards."
    if isinstance(column_names, list):
        raise TypeError("'column_names' must be a list of strings.")
    # Get valid columns 
    dtype_dict = get_L0_dtype(sensor_name)
    valid_columns = list(dtype_dict)
    valid_columns = valid_columns + ['time']
    # --------------------------------------------
    # Create name sets
    column_names = set(column_names)
    valid_columns = set(valid_columns)  
    # --------------------------------------------
    # Raise warning if there are columns not respecting DISDRODB standards 
    unvalid_columns = list(column_names.difference(valid_columns))
    if len(unvalid_columns) > 0:
        print("The following columns do no met the DISDRODB standards: {unvalid_columns}.")
        print("Please remove such columns within the df_sanitizer_fun")
    # --------------------------------------------
    # Check time column is present 
    if 'time' not in column_names:
        print("Please be sure to create the 'time' column within the df_sanitizer_fun.")
        print("The 'time' column must be datetime with resolution in seconds (dtype='M8[s]').")
    # --------------------------------------------
    return None

    def check_L0_column_names(x):
        # TODO: 
        # check_L0_column_names(column_names, sensor_name) 
        # --> Move in for loop 
        # --> Print message with columns to be drop in df_sanitizer 
        # --> Print message of columns to be derived in df_sanitizer (i.e. time)
        pass
    
    
def print_df_first_n_rows(df, n=5, column_names=True):
    columns = list(df.columns)
    for i in range(len(df.columns)):
        if column_names:
            print(" - Column", i, "(", columns[i], "):")
        else:
            print(" - Column", i, ":")
        print("     ", df.iloc[0 : (n + 1), i].values)
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
        print("     ", df.iloc[0 : (n + 1), i].values)
    return None


def print_df_column_names(df):
    for i, column in enumerate(df.columns):
        print(" - Column", i, ":", column)
    return None


def print_valid_L0_column_names(sensor_name):
    print(list(get_L0_dtype(sensor_name)))
    return None


def _check_valid_column_index(column_idx, n_columns):
    if column_idx > (n_columns - 1):
        raise ValueError("'column_idx' must be between 0 and {}".format(n_columns - 1))
    if column_idx < 0:
        raise ValueError("'column_idx' must be between 0 and {}".format(n_columns - 1))


def _check_columns_indices(column_indices, n_columns):
    if not isinstance(column_indices, (int, list, type(None), slice)):
        raise TypeError(
            "'column_indices' must be an integer, a list of integers, or None."
        )
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
        print("     ", sorted(df[column].unique().tolist()))
    return None


def get_df_columns_unique_values_dict(df, column_indices=None, column_names=True):
    """Create a dictionary {column: unique values}"""
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
        d[key] = sorted(df[column].unique().tolist())
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
    indices = indices[np.isin(indices, indices_to_remove, invert=True)]
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
        tmp_df.columns = [""]
        print("     ", tmp_df)
    return None


def print_df_with_any_nan_rows(df):
    df_bool_is_nan = df.isnull()
    idx_nan_rows = df_bool_is_nan.any(axis=1)
    df_nan_rows = df.loc[idx_nan_rows]
    print_df_first_n_rows(df_nan_rows, n=len(df_nan_rows))


####--------------------------------------------------------------------------.
#### Character checks
def arr_has_constant_nchar(arr):
    arr = np.asarray(arr)
    if arr.dtype.char not in ["O", "U"]:
        raise TypeError("Expecting object (O) or string (U) dtype.")
    if arr.dtype.char == "O":
        arr = arr.astype(str)
    if arr.dtype.char != "U":
        raise TypeError("Expecting string (U) dtype.")
    # Get number of characters (include .)
    str_nchars = np.char.str_len(arr)
    str_nchars_unique = np.unique(str_nchars)
    if len(str_nchars_unique) != 1:
        return False  # raise ValueError("Non-unique string length !")
    else:
        return True


def str_is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def str_is_not_number(string):
    return not str_is_number(string)


def str_is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def str_has_decimal_digits(string):
    if len(string.split(".")) == 2:
        return True
    else:
        return False


def get_decimal_ndigits(string):
    if str_has_decimal_digits(string):
        return len(string.split(".")[1])
    else:
        return 0


def get_natural_ndigits(string):
    if str_is_integer(string):
        return len(string)
    if str_has_decimal_digits(string):
        return len(string.split(".")[0])
    else:
        return 0


def get_ndigits(string):
    if str_is_not_number(string):
        return 0
    if str_has_decimal_digits(string):
        return len(string) - 1  # remove .
    else:
        return len(string)


def get_nchar(string):
    return len(string)


def get_possible_keys(dict_options, desired_value):
    list_key_match = []
    for k, v in dict_options.items():
        if v == desired_value:
            list_key_match.append(k)
    set_key_match = set(list_key_match)
    return set_key_match


def search_possible_columns(string, sensor_name):
    dict_digits = get_field_ndigits_dict(sensor_name)
    dict_nchar_digits = get_field_nchar_dict(sensor_name)
    dict_decimal_digits = get_field_ndigits_decimals_dict(sensor_name)
    dict_natural_digits = get_field_ndigits_natural_dict(sensor_name)
    set_digits = get_possible_keys(dict_digits, get_ndigits(string))
    set_nchar = get_possible_keys(dict_nchar_digits, get_nchar(string))
    set_decimals = get_possible_keys(dict_decimal_digits, get_decimal_ndigits(string))
    set_natural = get_possible_keys(dict_natural_digits, get_natural_ndigits(string))
    possible_keys = set_digits.intersection(set_nchar, set_decimals, set_natural)
    possible_keys = list(possible_keys)
    return possible_keys


def infer_df_str_column_names(df, sensor_name, row_idx=1):
    dict_possible_columns = {}
    for i, column in enumerate(df.columns):
        print(i)
        # Get string array
        arr = df.iloc[:, i]
        arr = np.asarray(arr).astype(str)
        if not arr_has_constant_nchar(arr):
            print("Column", i, "has non-unique number of characters")
            continue
        # Subset a single string
        string = arr[row_idx]
        possible_columns = search_possible_columns(string, sensor_name=sensor_name)
        dict_possible_columns[i] = possible_columns
    return dict_possible_columns
