#!/usr/bin/env python3

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2023 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Useful tools helping in the implementation of the DISDRODB L0 readers."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from disdrodb.l0.standards import (
    allowed_l0_variables,
    get_field_nchar_dict,
    get_field_ndigits_decimals_dict,
    get_field_ndigits_dict,
    get_field_ndigits_natural_dict,
    get_l0a_dtype,
)

#### Printing tool


def _get_selected_column_names(df, column_indices=None):
    columns = list(df.columns)
    if column_indices is None:
        return list(range(len(columns))), columns
    column_indices = _check_columns_indices(column_indices, len(columns))
    columns = [columns[idx] for idx in column_indices]
    return column_indices, columns


def _check_valid_column_index(column_idx, n_columns):
    if column_idx > (n_columns - 1):
        raise ValueError(f"'column_idx' must be between 0 and {n_columns - 1}")
    if column_idx < 0:
        raise ValueError(f"'column_idx' must be between 0 and {n_columns - 1}")


def _check_columns_indices(column_indices, n_columns):
    if not isinstance(column_indices, (int, list, slice)):
        raise TypeError("'column_indices' must be an integer, a list of integers, or None.")
    if isinstance(column_indices, slice):
        start = column_indices.start
        stop = column_indices.stop
        step = column_indices.step
        step = 1 if step is None else step
        column_indices = list(range(start, stop, step))
    if isinstance(column_indices, list):
        _ = [_check_valid_column_index(idx, n_columns) for idx in column_indices]
    if isinstance(column_indices, int):
        _check_valid_column_index(column_indices, n_columns)
        column_indices = [column_indices]
    return column_indices


def print_df_column_names(df: pd.DataFrame) -> None:
    """Print dataframe columns names.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe.
    """
    for i, column in enumerate(df.columns):
        print(" - Column", i, ":", column)


def print_allowed_column_names(sensor_name: str) -> None:
    """Print valid columns names from the standard.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    from pprint import pprint

    pprint(allowed_l0_variables(sensor_name))


def _print_column_index(i, column_name, print_column_names):
    if print_column_names:
        print(f" - Column {i} ( {column_name} ):")
    else:
        print(f" - Column {i} :")


def _print_value(value):
    print(f"      {value}")


def print_df_with_any_nan_rows(df: pd.DataFrame) -> None:
    """Print empty rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    """
    df_bool_is_nan = df.isna()
    idx_nan_rows = df_bool_is_nan.any(axis=1)
    df_nan_rows = df.loc[idx_nan_rows]
    if df_nan_rows.size != 0:
        print_df_first_n_rows(df_nan_rows, n=len(df_nan_rows))
    else:
        print("The dataframe does not have nan values!")


def print_df_first_n_rows(df: pd.DataFrame, n: int = 5, print_column_names: bool = True) -> None:
    """Print the n first n rows dataframe by column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    n : int, optional
        Number of row. The default is 5.
    column_names : bool , optional
        If true columns name are printed, by default ``True``.
    """
    columns = list(df.columns)
    for i in range(len(df.columns)):
        _print_column_index(i, column_name=columns[i], print_column_names=print_column_names)
        _print_value(df.iloc[0 : (n + 1), i].to_numpy())


def print_df_random_n_rows(df: pd.DataFrame, n: int = 5, print_column_names: bool = True) -> None:
    """Print the content of the dataframe by column, randomly chosen.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe.
    n : int, optional
        The number of row to print. The default is 5.
    print_column_names : bool, optional
        If true, print the column names. The default value is ``True``.
    """
    columns = list(df.columns)
    df_sample = df.sample(n=n)
    for i in range(len(df_sample.columns)):
        row_content = df_sample.iloc[0 : (n + 1), i].to_numpy()
        _print_column_index(i, column_name=columns[i], print_column_names=print_column_names)
        _print_value(row_content)


def _print_df_summary(df, indices, columns, print_column_names):
    # Compute summary stats
    summary_stats = ["mean", "min", "25%", "50%", "75%", "max"]
    df_summary = df.describe()
    df_summary = df_summary.loc[summary_stats]
    # Print summary stats
    for i, column in zip(indices, columns):
        tmp_df = df_summary[[column]]
        tmp_df.columns = [""]
        _print_column_index(i, column_name=column, print_column_names=print_column_names)
        _print_value(tmp_df)


def print_df_summary_stats(
    df: pd.DataFrame,
    column_indices: Optional[Union[int, slice, list]] = None,
    print_column_names: bool = True,
):
    """Create a columns statistics summary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    column_indices : Union[int,slice,list], optional
        Column indices. If ``None``, select all columns.
    print_column_names : bool, optional
        If ``True``, print the column names. The default value is ``True``.

    Raises
    ------
    ValueError
        Error if columns types is not numeric.

    """
    # Define columns of interest
    _, columns_of_interest = _get_selected_column_names(df, column_indices)
    # Remove columns of dtype object or string
    indices_to_remove = np.where((df.dtypes == type(object)) | (df.dtypes == str))  # noqa
    indices = np.arange(0, len(df.columns))
    indices = indices[np.isin(indices, indices_to_remove, invert=True)]
    columns = df.columns[indices]
    if len(columns) == 0:
        raise ValueError("No numeric columns in the dataframe.")
    # Select only columns of interest
    idx_of_interest = np.where(np.isin(columns, columns_of_interest))[0]
    if len(idx_of_interest) == 0:
        raise ValueError("No numeric columns at the specified column_indices.")
    columns = columns[idx_of_interest]
    indices = indices[idx_of_interest]
    # Print summary stats
    _print_df_summary(df=df, indices=indices, columns=columns, print_column_names=print_column_names)


def get_unique_sorted_values(array):
    """Return unique sorted values.

    It deals with np.nan within an array of string by converting object dtype to str.
    """
    arr = np.asanyarray(array)
    if arr.dtype == object:
        arr = arr.astype(str)
    return np.unique(arr).tolist()


def print_df_columns_unique_values(
    df: pd.DataFrame,
    column_indices: Optional[Union[int, slice, list]] = None,
    print_column_names: bool = True,
) -> None:
    """Print columns' unique values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    column_indices : Union[int,slice,list], optional
        Column indices. If ``None``, select all columns.
    column_names : bool, optional
        If ``True``, print the column names. The default value is ``True``.

    """
    column_indices, columns = _get_selected_column_names(df, column_indices)
    # Printing
    for i, column in zip(column_indices, columns):
        _print_column_index(i, column_name=column, print_column_names=print_column_names)
        _print_value(get_unique_sorted_values(df[column]))


####--------------------------------------------------------------------------.
#### Utility


def get_df_columns_unique_values_dict(
    df: pd.DataFrame,
    column_indices: Optional[Union[int, slice, list]] = None,
    column_names: bool = True,
):
    """Create a dictionary {column: unique values}.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    column_indices : Union[int,slice,list], optional
        Column indices. If ``None``, select all columns.
    column_names : bool, optional
        If ``True``, the dictionary key are the column names. The default value is ``True``.

    """
    column_indices, columns = _get_selected_column_names(df, column_indices)
    # Create dictionary
    d = {}
    for i, column in zip(column_indices, columns):
        key = column if column_names else "Column " + str(i)
        d[key] = get_unique_sorted_values(df[column])
    # Return
    return d


####--------------------------------------------------------------------------.
#### Character checks


def str_is_number(string: str) -> bool:
    """Check if a string represents a number.

    Parameters
    ----------
    string : str
        Input string.


    Returns
    -------
    bool
        ``True`` if float.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def str_is_integer(string: str) -> bool:
    """Check if a string represent an integer.

    Parameters
    ----------
    string : str
        Input string.


    Returns
    -------
    bool
        ``True`` if integer.
    """
    try:
        int(string)
        return True
    except ValueError:
        return False


def str_has_decimal_digits(string: str) -> bool:
    """Check if a string has decimals.

    Parameters
    ----------
    string : str
        Input string.


    Returns
    -------
    bool
        True if string has digits.
    """
    return len(string.split(".")) == 2


def get_decimal_ndigits(string: str) -> int:
    """Get the number of decimal digits.

    Parameters
    ----------
    string : str
        Input string.

    Returns
    -------
    int
        The number of decimal digits.
    """
    if str_has_decimal_digits(string):
        return len(string.split(".")[1])
    return 0


def get_natural_ndigits(string: str) -> int:
    """Get the number of natural digits.

    Parameters
    ----------
    string : str
        Input string.

    Returns
    -------
    int
        The number of natural digits.
    """
    count_minus = int(string.startswith("-"))  # 0 if not start with -, else 1
    string = string.replace("-", "")
    if str_is_integer(string):
        return len(string) + count_minus
    if str_has_decimal_digits(string):
        return len(string.split(".")[0]) + count_minus
    return 0


def get_ndigits(string: str) -> int:
    """Get the number of total numeric digits.

    Parameters
    ----------
    string : str
        Input string

    Returns
    -------
    int
        The number of total digits.
    """
    if not str_is_number(string):
        return 0
    count_minus = int(string.startswith("-"))  # 0 if not start with -, else 1
    string = string.replace("-", "")
    if str_has_decimal_digits(string):
        return len(string) - 1 + count_minus  # remove .
    return len(string) + count_minus


def get_nchar(string: str) -> int:
    """Get the number of characters.

    Parameters
    ----------
    string : str
        Input string.

    Returns
    -------
    int
        The number of characters.
    """
    return len(string)


def _has_constant_characters(arr: np.array) -> bool:
    """Check if the content of an array has a constant number of characters.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to analyse.
        It converts numeric array to unicode before analyzing !

    Returns
    -------
    boolean
        ``True`` if the number of characters is constant.
        Empty array are considered constant !

    """
    arr = np.asarray(arr).astype(str)
    # Get number of characters (include .)
    str_nchars = np.char.str_len(arr)
    str_nchars_unique = np.unique(str_nchars)
    return len(str_nchars_unique) in [0, 1]


def _get_possible_keys(dict_options: dict, desired_value: str) -> set:
    """Get the possible keys from the input values.

    Parameters
    ----------
    dict_options : dict
        Input dictionary.
    desired_value : str
        Input value.

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


def _search_possible_columns(string: str, sensor_name: str) -> list:
    """Define possible columns.

    Parameters
    ----------
    string : str
        Input string.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    list
        List of possible columns.
    """
    dict_digits = get_field_ndigits_dict(sensor_name)
    dict_nchar_digits = get_field_nchar_dict(sensor_name)
    dict_decimal_digits = get_field_ndigits_decimals_dict(sensor_name)
    dict_natural_digits = get_field_ndigits_natural_dict(sensor_name)

    set_digits = _get_possible_keys(dict_digits, get_ndigits(string))
    set_nchar = _get_possible_keys(dict_nchar_digits, get_nchar(string))
    set_decimals = _get_possible_keys(dict_decimal_digits, get_decimal_ndigits(string))
    set_natural = _get_possible_keys(dict_natural_digits, get_natural_ndigits(string))
    possible_keys = set_digits.intersection(set_nchar, set_decimals, set_natural)
    possible_keys = list(possible_keys)

    return possible_keys


####--------------------------------------------------------------------------.
#### Infer column names and checks validity


def infer_column_names(df: pd.DataFrame, sensor_name: str, row_idx: int = 0):
    """Try to guess the dataframe columns names based on string characteristics.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to analyse.
    sensor_name : str
        name of the sensor.
    row_idx : int, optional
        The row index of the dataframe to use to infer the column names.
        The default row index is 0.

    Returns
    -------
    dict
        Dictionary with the keys being the column id and the values being the guessed column names
    """
    dict_possible_columns = {}
    for i, _ in enumerate(df.columns):
        # Get string array
        arr = df.iloc[:, i]
        arr = np.asarray(arr).astype(str)
        # Check is the array contains a constant number of character
        if not _has_constant_characters(arr):
            print(
                f"WARNING: The number of characters of column {i} values is not constant. "
                f"Column names are currently inferred using 'row_idx={row_idx}'.",
            )

        # Subset a single string
        string = arr[row_idx]

        # Try to guess the column
        possible_columns = _search_possible_columns(string, sensor_name=sensor_name)
        dict_possible_columns[i] = possible_columns

    return dict_possible_columns


def check_column_names(column_names: list, sensor_name: str) -> None:
    """Checks that the column names respects DISDRODB standards.

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
    dtype_dict = get_l0a_dtype(sensor_name)
    valid_columns = list(dtype_dict)
    valid_columns = [*valid_columns, "time"]
    # --------------------------------------------
    # Create name sets
    column_names = set(column_names)
    valid_columns = set(valid_columns)
    # --------------------------------------------
    # Raise warning if there are columns not respecting DISDRODB standards
    invalid_columns = list(column_names.difference(valid_columns))
    if len(invalid_columns) > 0:
        print(f"The following columns do no met the DISDRODB standards: {invalid_columns}.")
        print("Please remove such columns in the reader function !")
    # --------------------------------------------
    # Check time column is present
    if "time" not in column_names:
        print("Please be sure to create the 'time' column within the reader function !")
        print("The 'time' column must be datetime with resolution in seconds (dtype='M8[s]').")
