#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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
"""Functions to process raw text files into DISDRODB L0A Apache Parquet."""

# -----------------------------------------------------------------------------.
import os
import inspect
import logging
import pandas as pd
import numpy as np
from typing import Union
from disdrodb.l0.standards import (
    get_l0a_dtype,
    get_nan_flags_dict,
    get_data_range_dict,
    get_valid_values_dict,
)
from disdrodb.l0.check_standards import check_l0a_column_names, check_l0a_standards
from disdrodb.l0.io import _remove_if_exists, _create_directory
from disdrodb.l0.l0b_processing import infer_split_str

# Logger
from disdrodb.utils.logger import (
    log_info,
    log_warning,
    log_error,
    log_debug,
)

logger = logging.getLogger(__name__)

pd.set_option("mode.chained_assignment", None)  # Avoid SettingWithCopyWarning
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters


####---------------------------------------------------------------------------.
#### Raw file readers


def preprocess_reader_kwargs(reader_kwargs: dict) -> dict:
    """Preprocess arguments required to read raw text file into Pandas.

    Parameters
    ----------
    reader_kwargs : dict
        Initial parameter dictionary.

    Returns
    -------
    dict
        Parameter dictionary that matches either Pandas or Dask.
    """
    # Check delimiter is specified !
    if "delimiter" not in reader_kwargs:
        raise ValueError("The 'delimiter' key must be specified in reader_kwargs dictionary!")

    # Remove dtype key
    # - The dtype is enforced to be 'object' in the read function !
    reader_kwargs.pop("dtype", None)

    # Preprocess the reader_kwargs
    reader_kwargs = reader_kwargs.copy()

    # Remove kwargs expected by dask dataframe read_csv
    reader_kwargs.pop("blocksize", None)
    reader_kwargs.pop("assume_missing", None)

    return reader_kwargs


def read_raw_data(
    filepath: str,
    column_names: list,
    reader_kwargs: dict,
) -> pd.DataFrame:
    """Read raw data into a dataframe.

    Parameters
    ----------
    filepath : str
        Raw file path.
    column_names : list
        Column names.
    reader_kwargs : dict
        Pandas pd.read_csv arguments.

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe.
    """
    # Preprocess reader_kwargs
    reader_kwargs = preprocess_reader_kwargs(reader_kwargs)

    # Enforce all raw files columns with dtype = 'object'
    dtype = "object"

    # Try to read the data
    try:
        df = pd.read_csv(filepath, names=column_names, dtype=dtype, **reader_kwargs)
    except pd.errors.EmptyDataError:
        msg = f" - Is empty, skip file: {filepath}"
        log_warning(logger=logger, msg=msg, verbose=False)
        pass

    # Return dataframe
    return df


####---------------------------------------------------------------------------.
#### L0A checks and homogenization


def _check_df_sanitizer_fun(df_sanitizer_fun):
    """Check the argument of df_sanitizer_fun is only df."""
    if df_sanitizer_fun is None:
        return None
    if not callable(df_sanitizer_fun):
        raise ValueError("'df_sanitizer_fun' must be a function.")
    if not np.all(np.isin(inspect.getfullargspec(df_sanitizer_fun).args, ["df"])):
        raise ValueError("The `df_sanitizer_fun` must have only `df` as input argument!")


def _check_not_empty_dataframe(df, verbose=False):
    if len(df.index) == 0:
        msg = " - The file is empty and has been skipped."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)


def _check_matching_column_number(df, column_names, verbose=False):
    n_columns = len(df.columns)
    n_expected_columns = len(column_names)
    if n_columns != n_expected_columns:
        msg = f" - The dataframe has {n_columns} columns, while {n_expected_columns} are expected !."
        log_error(logger, msg, verbose)
        raise ValueError(msg)


def remove_rows_with_missing_time(df: pd.DataFrame, verbose: bool = False):
    """Remove dataframe rows where the "time" is NaT.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    verbose : bool
        Wheter to verbose the processing.

    Returns
    -------
    pd.DataFrame
        Dataframe with valid timesteps.
    """
    # Get the number of rows of the dataframe
    n_rows = len(df)
    # Drop rows with "time" nat values
    df = df.dropna(subset="time", axis=0)
    # If no valid timesteps, raise error
    if len(df.index) == 0:
        msg = " - There are not valid timestep."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    # Otherwise, report the number of unvalid timesteps
    n_unvalid_timesteps = n_rows - len(df)
    if n_unvalid_timesteps > 0:
        msg = f" - {n_unvalid_timesteps} rows had unvalid timesteps and were discarded."
        log_warning(logger=logger, msg=msg, verbose=verbose)
    return df


def remove_duplicated_timesteps(df: pd.DataFrame, verbose: bool = False):
    """Remove duplicated timesteps.

    It keep only the first timestep occurence !

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    verbose : bool
        Wheter to verbose the processing.

    Returns
    -------
    pd.DataFrame
        Dataframe with valid unique timesteps.
    """
    values, counts = np.unique(df["time"], return_counts=True)
    idx_duplicates = np.where(counts > 1)[0]
    values_duplicates = values[idx_duplicates].astype("M8[s]")
    # If there are duplicated timesteps
    if len(values_duplicates) > 0:
        # Drop duplicated timesteps (keeping the first occurence)
        df = df.drop_duplicates(subset="time", keep="first")
        # Report the values of duplicated timesteps
        msg = (
            f" - The following timesteps occured more than once: {values_duplicates}. Only the first occurence"
            " selected."
        )
        log_warning(logger=logger, msg=msg, verbose=verbose)
    return df


def drop_timesteps(df, timesteps):
    """Drop problematic time steps."""
    df = df[~df["time"].isin(timesteps)]
    # Check there are row left
    if len(df) == 0:
        msg = "No rows left after removing problematic timesteps. Maybe you need to adjust the issue YAML file."
        log_warning(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    return df


def drop_time_periods(df, time_periods):
    """Drop problematic time_period."""
    for time_period in time_periods:
        if len(df) > 0:
            start_time = time_period[0]
            end_time = time_period[1]
            df = df[(df["time"] < start_time) | (df["time"] > end_time)]
    # Check there are row left
    if len(df) == 0:
        msg = "No rows left after removing problematic time_periods. Maybe you need to adjust the issue YAML file."
        log_warning(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)

    return df


def remove_issue_timesteps(df, issue_dict, verbose=False):
    """Drop dataframe rows with timesteps listed in the issue dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    issue_dict : dict
        Issue dictionary

    Returns
    -------
    pd.DataFrame
        Dataframe with problematic timesteps removed.

    """
    # Retrieve number of initial rows
    n_initial_rows = len(df)

    # Retrieve timesteps and time_periods
    timesteps = issue_dict.get("timesteps", None)
    time_periods = issue_dict.get("time_periods", None)

    # Drop rows of specified timesteps
    if timesteps:
        df = drop_timesteps(df=df, timesteps=timesteps)

    # Drop rows within specified time_period
    if time_periods:
        df = drop_time_periods(df, time_periods=time_periods)

    # Report number fo dropped rows
    n_rows_dropped = n_initial_rows - len(df)
    if n_rows_dropped > 0:
        msg = f"{n_rows_dropped} rows were dropped following the issue YAML file content."
        log_info(logger=logger, msg=msg, verbose=verbose)

    return df


def cast_column_dtypes(df: pd.DataFrame, sensor_name: str, verbose: bool = False) -> pd.DataFrame:
    """Convert 'object' dataframe columns into DISDRODB L0A dtype standards.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.

    Returns
    -------
    pd.DataFrame
        Dataframe with corrected columns types.
    """

    # Cast dataframe to dtypes
    dtype_dict = get_l0a_dtype(sensor_name)
    # Ensure time column is saved with seconds resolution
    dtype_dict["time"] = "M8[s]"
    # Add latitude, longitude and elevation for mobile disdrometers
    dtype_dict["latitude"] = "float64"
    dtype_dict["longitude"] = "float64"
    dtype_dict["altitude"] = "float64"
    # Get dataframe column names
    columns = list(df.columns)
    # Cast dataframe columns
    for column in columns:
        try:
            df[column] = df[column].astype(dtype_dict[column])
        except ValueError as e:
            msg = f"ValueError: The column {column} has {e}"
            log_error(logger=logger, msg=msg, verbose=False)
            raise ValueError(msg)
    return df


def coerce_corrupted_values_to_nan(df: pd.DataFrame, sensor_name: str, verbose: bool = False) -> pd.DataFrame:
    """Coerce corrupted values in dataframe numeric columns to np.nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.

    Returns
    -------
    pd.DataFrame
        Dataframe with string columns without corrupted values.
    """
    # Cast dataframe to dtypes
    dtype_dict = get_l0a_dtype(sensor_name)

    # Get string columns
    numeric_columns = [k for k, dtype in dtype_dict.items() if "float" in dtype or "int" in dtype]

    # Get dataframe column names
    columns = list(df.columns)

    # Cast dataframe columns
    for column in columns:
        if column in numeric_columns:
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            except AttributeError:
                msg = f"The column {column} is not a numeric column."
                log_error(logger=logger, msg=msg, verbose=False)
                raise ValueError(msg)
    return df


def strip_string_spaces(df: pd.DataFrame, sensor_name: str, verbose: bool = False) -> pd.DataFrame:
    """Strip leading/trailing spaces from dataframe string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.

    Returns
    -------
    pd.DataFrame
        Dataframe with string columns without leading/trailing spaces.
    """
    # Cast dataframe to dtypes
    dtype_dict = get_l0a_dtype(sensor_name)

    # Get string columns
    string_columns = [k for k, dtype in dtype_dict.items() if dtype == "str"]

    # Get dataframe column names
    columns = list(df.columns)
    # Cast dataframe columns
    for column in columns:
        if column in string_columns:
            try:
                df[column] = df[column].str.strip()
            except AttributeError:
                msg = f"AttributeError: The column {column} is not a string/object dtype."
                log_error(logger=logger, msg=msg, verbose=False)
                raise AttributeError(msg)
    return df


def _strip_delimiter(string):
    if not isinstance(string, str):
        return string
    split_str = infer_split_str(string=string)
    string = string.strip(split_str)
    return string


def strip_delimiter_from_raw_arrays(df):
    """Remove the first and last delimiter occurence from the raw array fields."""
    # Possible fields
    possible_fields = [
        "raw_drop_number",
        "raw_drop_concentration",
        "raw_drop_average_velocity",
    ]
    available_fields = list(df.columns[np.isin(df.columns, possible_fields)])
    # Loop over the fields and strip away the delimiter
    for field in available_fields:
        df[field] = df[field].apply(_strip_delimiter)
    # Return the dataframe
    return df


def _is_not_corrupted(string):
    """Check if the raw array is corrupted."""
    if not isinstance(string, str):
        return False
    split_str = infer_split_str(string=string)
    list_values = string.split(split_str)
    values = pd.to_numeric(list_values, errors="coerce")
    return ~np.any(np.isnan(values))


def remove_corrupted_rows(df):
    """Remove corrupted rows by checking conversion of raw fields to numeric.

    Note: The raw array must be stripped away from delimiter at start and end !
    """
    # Possible fields
    possible_fields = [
        "raw_drop_number",
        "raw_drop_concentration",
        "raw_drop_average_velocity",
    ]
    available_fields = list(df.columns[np.isin(df.columns, possible_fields)])
    # Loop over the fields and remove corrupted ones
    for field in available_fields:
        if len(df) != 0:
            df = df[df[field].apply(_is_not_corrupted)]
    # Check if there are rows left
    if len(df) == 0:
        raise ValueError("No remaining rows after data corruption checks.")
    # If only one row available, raise also error
    if len(df) == 1:
        raise ValueError("Only 1 row remains after data corruption checks. Check the file.")
    # Return the dataframe
    return df


def replace_nan_flags(df, sensor_name, verbose):
    """Set values corresponding to nan_flags to np.nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.

    Returns
    -------
    pd.DataFrame
        Dataframe without nan_flags values.
    """
    # Get dictionary of nan flags
    dict_nan_flags = get_nan_flags_dict(sensor_name)
    # Loop over the needed variable, and replace nan_flags values with np.nan
    for var, nan_flags in dict_nan_flags.items():
        # If the variable is in the dataframe
        if var in df:
            # Get array with occurence of nan_flags
            is_a_nan_flag = df[var].isin(nan_flags)
            # If nan_flags values are present, replace with np.nan
            n_nan_flags_values = np.sum(is_a_nan_flag)
            if n_nan_flags_values > 0:
                msg = f"In variable {var}, {n_nan_flags_values} values were nan_flags and were replaced to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                df[var][is_a_nan_flag] = np.nan
    # Return dataframe
    return df


def set_nan_outside_data_range(df, sensor_name, verbose):
    """Set values outside the data range as np.nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.

    Returns
    -------
    pd.DataFrame
        Dataframe without values outside the expected data range.
    """
    # Get dictionary of data_range
    dict_data_range = get_data_range_dict(sensor_name)
    # Loop over the variable with a defined data_range
    for var, data_range in dict_data_range.items():
        # If the variable is in the dataframe
        if var in df:
            # Get min and max value
            min_val = data_range[0]
            max_val = data_range[1]
            # Check within data range or already np.nan
            is_valid = (df[var] >= min_val) & (df[var] <= max_val) | df[var].isna()
            # If there are values outside the data range, set to np.nan
            n_unvalid = np.sum(~is_valid)
            if n_unvalid > 0:
                msg = f"{n_unvalid} {var} values were outside the data range and were set to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                df[var] = df[var].where(is_valid)  # set not valid to np.nan

    # Return dataframe
    return df


def set_nan_unvalid_values(df, sensor_name, verbose):
    """Set unvalid (class) values to np.nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.

    Returns
    -------
    pd.DataFrame
        Dataframe without unvalid values.
    """
    # Get dictionary of valid values
    dict_valid_values = get_valid_values_dict(sensor_name)
    # Loop over the variable with a defined data_range
    for var, valid_values in dict_valid_values.items():
        # If the variable is in the dataframe
        if var in df:
            # Get array with occurence of correct values (or already np.nan)
            is_valid_values = df[var].isin(valid_values) | df[var].isna()
            # If unvalid values are present, replace with np.nan
            n_unvalid_values = np.sum(~is_valid_values)
            if n_unvalid_values > 0:
                msg = f"{n_unvalid_values} {var} values were unvalid and were replaced to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                df[var] = df[var].where(is_valid_values)  # set not valid to np.nan

    # Return dataframe
    return df


def process_raw_file(
    filepath,
    column_names,
    reader_kwargs,
    df_sanitizer_fun,
    sensor_name,
    verbose=True,
    issue_dict={},
):
    """Read and parse a raw text files into a L0A dataframe.

    Parameters
    ----------
    filepath : str
        File path
    column_names : list
        Columns names.
    reader_kwargs : dict
         Pandas `read_csv` arguments.
    df_sanitizer_fun : object, optional
        Sanitizer function to format the datafame.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.
        The default is True
    issue_dict : dict
        Issue dictionary providing information on timesteps to remove.
        The default is an empty dictionary {}.
        Valid issue_dict key are 'timesteps' and 'time_periods'.
        Valid issue_dict values are list of datetime64 values (with second accuracy).
        To correctly format and check the validity of the issue_dict, use
        the disdrodb.l0.issue.check_issue_dict function.

    Returns
    -------
    pd.DataFrame
        Dataframe
    """
    _check_df_sanitizer_fun(df_sanitizer_fun)

    # Read the data
    df = read_raw_data(
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
    )

    # - Check if file empty
    _check_not_empty_dataframe(df=df, verbose=verbose)

    # - Check dataframe column number matches columns_names
    _check_matching_column_number(df, column_names, verbose=False)

    # - Sanitize the dataframe with a custom function
    if df_sanitizer_fun is not None:
        df = df_sanitizer_fun(df)

    # - Remove rows with time NaT
    df = remove_rows_with_missing_time(df, verbose=verbose)

    # - Remove duplicated timesteps
    df = remove_duplicated_timesteps(df, verbose=verbose)

    # - Filter out problematic tiemsteps reported in the issue YAML file
    df = remove_issue_timesteps(df, issue_dict=issue_dict, verbose=verbose)

    # - Coerce numeric columns corrupted values to np.nan
    df = coerce_corrupted_values_to_nan(df, sensor_name=sensor_name, verbose=verbose)

    # - Strip trailing/leading space from string columns
    df = strip_string_spaces(df, sensor_name=sensor_name, verbose=verbose)

    # - Strip first and last delimiter from the raw arrays
    df = strip_delimiter_from_raw_arrays(df)

    # - Remove corrupted rows
    df = remove_corrupted_rows(df)

    # - Cast dataframe to dtypes
    df = cast_column_dtypes(df, sensor_name=sensor_name, verbose=verbose)

    # - Replace nan flags values with np.nans
    df = replace_nan_flags(df, sensor_name=sensor_name, verbose=verbose)

    # - Set values outside the data range to np.nan
    df = set_nan_outside_data_range(df, sensor_name=sensor_name, verbose=verbose)

    # - Replace unvalid values with np.nan
    df = set_nan_unvalid_values(df, sensor_name=sensor_name, verbose=verbose)

    # ------------------------------------------------------.
    # - Check column names agrees to DISDRODB standards
    check_l0a_column_names(df, sensor_name=sensor_name)

    # - Check the dataframe respects the DISDRODB standards
    check_l0a_standards(df=df, sensor_name=sensor_name, verbose=verbose)

    # ------------------------------------------------------.
    # Return the L0A dataframe
    return df


####---------------------------------------------------------------------------.
#### L0A Apache Parquet Writer


def write_l0a(
    df: pd.DataFrame,
    fpath: str,
    force: bool = False,
    verbose: bool = False,
):
    """Save the dataframe into an Apache Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fpath : str
        Output file path.
    force : bool, optional
        Whether to overwrite existing data.
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories. This is the default.
    verbose : bool, optional
        Wheter to verbose the processing.
        The default is False.

    Raises
    ------
    ValueError
        The input dataframe can not be written as an Apache Parquet file.
    NotImplementedError
        The input dataframe can not be processed.
    """

    # -------------------------------------------------------------------------.
    # Create station directory if does not exist
    _create_directory(os.path.dirname(fpath))

    # Check if the file already exists
    # - If force=True --> Remove it
    # - If force=False --> Raise error
    _remove_if_exists(fpath, force=force)

    # -------------------------------------------------------------------------.
    # Define writing options
    compression = "snappy"  # 'gzip', 'brotli, 'lz4', 'zstd'
    row_group_size = 100000
    engine = "pyarrow"
    # -------------------------------------------------------------------------.
    # Save dataframe to Apache Parquet
    try:
        df.to_parquet(
            fpath,
            engine=engine,
            compression=compression,
            row_group_size=row_group_size,
        )
        msg = f"The Pandas Dataframe has been written as an Apache Parquet file to {fpath}."
        log_info(logger=logger, msg=msg, verbose=False)
    except Exception as e:
        msg = f" - The Pandas DataFrame cannot be written as an Apache Parquet file. The error is: \n {e}."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    # -------------------------------------------------------------------------.
    return None


####---------------------------------------------------------------------------.
#### L0A Utility


def concatenate_dataframe(list_df: list, verbose: bool = False) -> pd.DataFrame:
    """Concatenate a list of dataframes.

    Parameters
    ----------
    list_df : list
        List of dataframes.
    verbose : bool, optional
        If True, print messages.
        If False, no print.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe.

    Raises
    ------
    ValueError
        Concatenation can not be done.
    """
    # Check if something to concatenate
    if len(list_df) == 1:
        df = list_df[0]
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Only pd.DataFrame objects are valid.")
        return df

    # Log
    msg = " - Concatenation of dataframes started."
    log_info(logger, msg, verbose)

    # Concatenate the dataframe
    try:
        df = pd.concat(list_df, axis=0, ignore_index=True)

        # Drop duplicated values
        df = df.drop_duplicates(subset="time")

        # Sort by increasing time
        df = df.sort_values(by="time")

    except (AttributeError, TypeError) as e:
        msg = f" - Can not concatenate the files. \n Error: {e}"
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)

    # Log
    msg = " - Concatenation of dataframes has finished."
    log_info(logger, msg, verbose)

    # Return dataframe
    return df


def read_raw_file_list(
    file_list: Union[list, str],
    column_names: list,
    reader_kwargs: dict,
    sensor_name: str,
    verbose: bool,
    df_sanitizer_fun: object = None,
) -> pd.DataFrame:
    """Read and parse a list for raw files into a dataframe.

    Parameters
    ----------
    file_list : Union[list,str]
        File(s) path(s)
    column_names : list
        Columns names.
    reader_kwargs : dict
         Pandas `read_csv` arguments.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.
    df_sanitizer_fun : object, optional
        Sanitizer function to format the datafame.

    Returns
    -------
    pd.DataFrame
        Dataframe

    Raises
    ------
    ValueError
        Input parameters can not be used or the raw file can not be processed.

    """

    # ------------------------------------------------------.
    # Check input list
    if isinstance(file_list, str):
        file_list = [file_list]
    if len(file_list) == 0:
        raise ValueError("'file_list' must contains at least 1 filepath.")

    # ------------------------------------------------------.
    ### - Loop over all raw files
    n_files = len(file_list)
    processed_file_counter = 0
    list_skipped_files_msg = []
    list_df = []
    for filepath in file_list:
        try:
            # Try to process a raw file
            df = process_raw_file(
                filepath=filepath,
                column_names=column_names,
                reader_kwargs=reader_kwargs,
                df_sanitizer_fun=df_sanitizer_fun,
                sensor_name=sensor_name,
                verbose=verbose,
            )

            # Append dataframe to the list
            list_df.append(df)

            # Update the logger
            processed_file_counter += 1
            msg = f"{processed_file_counter} / {n_files} processed successfully. File name: {filepath}"
            log_debug(logger=logger, msg=msg, verbose=verbose)

        # If processing of raw file fails
        except Exception as e:
            # Update the logger
            msg = f" - {filepath} has been skipped. \n -- The error is: {e}."
            log_warning(logger=logger, msg=msg, verbose=verbose)
            list_skipped_files_msg.append(msg)

    # Update logger
    msg = f" - {len(list_skipped_files_msg)} of {n_files} have been skipped."
    log_info(logger=logger, msg=msg, verbose=verbose)
    logger.info("---")
    logger.info(msg)
    logger.info("---")

    ##----------------------------------------------------------------.
    #### - Concatenate the dataframe
    if len(list_df) == 0:
        raise ValueError(f"No dataframe to return. Impossible to parse {file_list}.")
    df = concatenate_dataframe(list_df, verbose=verbose)

    # - Remove rows with duplicate timestep (keep the first)
    df = df.drop_duplicates(subset=["time"], keep="first")

    # ------------------------------------------------------.
    # Return the dataframe
    return df
