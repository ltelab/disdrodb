#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:56:38 2022

@author: ghiggi
"""
import numpy as np
import pandas as pd
from typing import Union
from disdrodb.L0.standards import get_L0A_dtype
from disdrodb.L0.check_standards import (
    get_field_nchar_dict,
    get_field_ndigits_dict,
    get_field_ndigits_decimals_dict,
    get_field_ndigits_natural_dict,
)


def check_column_names(column_names: list, sensor_name: str) -> None:
    """Checks that the columnn names respects DISDRODB standards.

    Parameters
    ----------
    column_names : list
        List of columns names.
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    TypeError
        Error if some columns do not meet the DISDRODB standards.
    """

    if not isinstance(column_names, list):
        raise TypeError("'column_names' must be a list of strings.")
    # Get valid columns
    dtype_dict = get_L0A_dtype(sensor_name)
    valid_columns = list(dtype_dict)
    valid_columns = valid_columns + ["time"]
    # --------------------------------------------
    # Create name sets
    column_names = set(column_names)
    valid_columns = set(valid_columns)
    # --------------------------------------------
    # Raise warning if there are columns not respecting DISDRODB standards
    invalid_columns = list(column_names.difference(valid_columns))
    if len(invalid_columns) > 0:
        print(
            f"The following columns do no met the DISDRODB standards: {invalid_columns}."
        )
        print("Please remove such columns within the df_sanitizer_fun")
    # --------------------------------------------
    # Check time column is present
    if "time" not in column_names:
        print("Please be sure to create the 'time' column within the df_sanitizer_fun.")
        print(
            "The 'time' column must be datetime with resolution in seconds (dtype='M8[s]')."
        )
    # --------------------------------------------
    return None

    def check_L0_column_names(x):
        # TODO:
        # check_L0_column_names(column_names, sensor_name)
        # --> Move in for loop
        # --> Print message with columns to be drop in df_sanitizer
        # --> Print message of columns to be derived in df_sanitizer (i.e. time)
        pass


def print_df_first_n_rows(
    df: pd.DataFrame, n: int = 5, column_names: bool = True
) -> None:
    """Print the n first n rows dataframe by column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    n : int, optional
        Number of row, by default 5
    column_names : bool , optional
        If true columns name are printed, by default True
    """
    columns = list(df.columns)
    for i in range(len(df.columns)):
        if column_names:
            print(" - Column", i, "(", columns[i], "):")
        else:
            print(" - Column", i, ":")
        print("     ", df.iloc[0 : (n + 1), i].values)
    return None


def print_df_random_n_rows(
    df: pd.DataFrame, n: int = 5, with_column_names: bool = True
) -> None:
    """Print the content of the dataframe by column, randomly chosen

    Parameters
    ----------
    df : dataframe
        The dataframe
    n : int, optional
        The number of row to print, by default 5
    with_column_names : bool, optional
        If true, print the column name, by default True

    Returns
    -------
    None
        Nothing
    """

    df = df.copy()
    df = df.sample(n=n)

    if with_column_names:
        columns = list(df.columns)

    for i in range(len(df.columns)):
        row_content = df.iloc[0 : (n + 1), i].values
        if with_column_names:
            columns = list(df.columns)
            print(f"- Column {i} ({columns[i]}) : {row_content}")
        else:
            print(f"- Column {i} : {row_content}")

    return None


def print_df_column_names(df: pd.DataFrame) -> None:
    """Print dataframe columns names

    Parameters
    ----------
    df : dataframe
        The dataframe

    Returns
    -------
    None
        Nothing
    """
    for i, column in enumerate(df.columns):
        print(" - Column", i, ":", column)
    return None


def print_valid_L0_column_names(sensor_name: str) -> None:
    """Print valid columns names from the standard.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    print(list(get_L0A_dtype(sensor_name)))
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


def print_df_columns_unique_values(
    df: pd.DataFrame,
    column_indices: Union[int, slice, list] = None,
    column_names: bool = True,
) -> None:
    """Print columns' unique values

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column_indices : Union[int,slice,list], optional
        column indices
    column_names : bool, optional
        If true, print the column name, by default True

    """
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


def get_df_columns_unique_values_dict(
    df: pd.DataFrame,
    column_indices: Union[int, slice, list] = None,
    column_names: bool = True,
):
    """Create a dictionary {column: unique values}

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column_indices : Union[int,slice,list], optional
        column indices
    column_names : bool, optional
        If true, print the column name, by default True

    """

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


def print_df_summary_stats(
    df: pd.DataFrame,
    column_indices: Union[int, slice, list] = None,
    column_names: bool = True,
):
    """Create a columns statistics summary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column_indices : Union[int,slice,list], optional
        column indices
    column_names : bool, optional
        If true, print the column name, by default True

    Raises
    ------
    ValueError
        Error if columns types is not numeric.

    """
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


def print_df_with_any_nan_rows(df: pd.DataFrame) -> None:
    """Print empty rows

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    """
    df_bool_is_nan = df.isnull()
    idx_nan_rows = df_bool_is_nan.any(axis=1)
    df_nan_rows = df.loc[idx_nan_rows]
    print_df_first_n_rows(df_nan_rows, n=len(df_nan_rows))


####--------------------------------------------------------------------------.
#### Character checks
def arr_has_constant_nchar(arr: np.array) -> bool:
    """Check if the content of an array has a constant number of characters

    Parameters
    ----------
    arr : numpy.ndarray
        The array to analyse

    Returns
    -------
    booleen
        True if the number of character is constant

    """
    arr = np.asarray(arr)
    # Get unique character code
    unique_character_code = arr.dtype.char

    if unique_character_code == "O":  # If (Python) objects
        arr = arr.astype(str)
    elif unique_character_code != "U":  # or if not Unicode string
        raise TypeError("Expecting object (O) or string (U) dtype.")

    # Get number of characters (include .)
    str_nchars = np.char.str_len(arr)
    str_nchars_unique = np.unique(str_nchars)

    if len(str_nchars_unique) != 1:
        return False  # raise ValueError("Non-unique string length !")
    else:
        return True


def str_is_number(string: str) -> bool:
    """Check if a string is numeric

    Parameters
    ----------
    string : Input string


    Returns
    -------
    bool
        True if float.
    """

    try:
        float(string)
        return True
    except ValueError:
        return False


def str_is_not_number(string: str) -> bool:
    """Check if a string is not numeric

    Parameters
    ----------
    string : Input string


    Returns
    -------
    bool
        True if not float.
    """
    return not str_is_number(string)


def str_is_integer(string: str) -> bool:
    """Check if a string is an integer

    Parameters
    ----------
    string : Input string


    Returns
    -------
    bool
        True if integer.
    """
    try:
        int(string)
        return True
    except ValueError:
        return False


def str_has_decimal_digits(string: str) -> bool:
    """Check if a string has decimals

    Parameters
    ----------
    string :
        Input string


    Returns
    -------
    bool
        True if sting has digits.
    """
    if len(string.split(".")) == 2:
        return True
    else:
        return False


def get_decimal_ndigits(string: str) -> int:
    """Get the decimal number of digit.

    Parameters
    ----------
    string : str
        Input string

    Returns
    -------
    int
        The number of digit.
    """
    if str_has_decimal_digits(string):
        return len(string.split(".")[1])
    else:
        return 0


def get_natural_ndigits(string: str) -> int:
    """Get the natural number of digit.

    Parameters
    ----------
    string : str
        Input string

    Returns
    -------
    int
        The number of digit.
    """
    if str_is_integer(string):
        return len(string)
    if str_has_decimal_digits(string):
        return len(string.split(".")[0])
    else:
        return 0


def get_ndigits(string: str) -> int:
    """Get the number of digit.

    Parameters
    ----------
    string : str
        Input string

    Returns
    -------
    int
        Number of digit
    """

    if str_is_not_number(string):
        return 0
    if str_has_decimal_digits(string):
        return len(string) - 1  # remove .
    else:
        return len(string)


def get_nchar(string: str) -> int:
    """Get the number of charactar.

    Parameters
    ----------
    string : str
        Input string

    Returns
    -------
    int
        Number of charactar
    """
    return len(string)


def get_possible_keys(dict_options: dict, desired_value: str) -> set:
    """Get the possible keys from the input values

    Parameters
    ----------
    dict_options : dict
        Input dictionnary
    desired_value : str
        Input value

    Returns
    -------
    set
        Keys that the value matches the desired input value.
    """

    list_key_match = []
    for k, v in dict_options.items():
        if v == desired_value:
            list_key_match.append(k)
    set_key_match = set(list_key_match)
    return set_key_match


def search_possible_columns(string: str, sensor_name: str) -> list:
    """Define possible column

    Parameters
    ----------
    string : str
        Inpur string
    sensor_name : str
        Name of the sensor

    Returns
    -------
    list
        list of possible columns
    """
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


def infer_df_str_column_names(df: pd.DataFrame, sensor_name: str, row_idx: int = 1):
    """Try to guess the columns names base on sting patterns.

    Parameters
    ----------
    df : numpy.ndarray
        The array to analyse
    sensor_name : str
        name of the sensor
    row_idx : int, optional
        The row ID of the array, by default 1

    Returns
    -------
    dict
        Dictionary with the keys being the column id and the values being the guessed column names
    """
    dict_possible_columns = {}
    for i, column in enumerate(df.columns):

        # Get string array
        arr = df.iloc[:, i]
        arr = np.asarray(arr).astype(str)

        # check is the array contains a constant number of character
        if not arr_has_constant_nchar(arr):
            print("Column", i, "has non-unique number of characters")

        # Subset a single string
        string = arr[row_idx]

        # Try to guess the column
        possible_columns = search_possible_columns(string, sensor_name=sensor_name)
        dict_possible_columns[i] = possible_columns

    return dict_possible_columns
