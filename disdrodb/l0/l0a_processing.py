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
"""Functions to process raw text files into DISDRODB L0A Apache Parquet."""


import logging
import os
from typing import Union

import numpy as np
import pandas as pd

from disdrodb.l0.check_standards import check_l0a_column_names, check_l0a_standards
from disdrodb.l0.l0b_processing import infer_split_str
from disdrodb.l0.standards import (
    get_data_range_dict,
    get_l0a_dtype,
    get_nan_flags_dict,
    get_valid_values_dict,
)
from disdrodb.utils.directories import create_directory, remove_if_exists

# Logger
from disdrodb.utils.logger import (
    log_error,
    log_info,
    log_warning,
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


def check_matching_column_number(df, column_names):
    """Check the number of columns in the dataframe matches the length of column names."""
    n_columns = len(df.columns)
    n_expected_columns = len(column_names)
    if n_columns != n_expected_columns:
        msg = f"The dataframe has {n_columns} columns, while {n_expected_columns} are expected !."
        raise ValueError(msg)


def read_raw_text_file(
    filepath: str,
    column_names: list,
    reader_kwargs: dict,
    logger=None,  # noqa
) -> pd.DataFrame:
    """Read a raw file into a dataframe.

    Parameters
    ----------
    filepath : str
        Raw file path.
    column_names : list
        Column names.
    reader_kwargs : dict
        Pandas ``pd.read_csv`` arguments.
    logger : logging.Logger
        Logger object.
        The default is ``None``.
        If ``None``, the logger is created using the module name.
        If ``logger`` is passed, it will be used to log messages.

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
        msg = f"The following file is empty: {filepath}"
        raise ValueError(msg)

    # Check the dataframe is not empty
    if len(df.index) == 0:
        msg = f"The following file is empty: {filepath}"
        raise ValueError(msg)

    # Check dataframe column number matches columns_names
    if column_names is not None:
        check_matching_column_number(df, column_names)

    # Return dataframe
    return df


####---------------------------------------------------------------------------.
#### L0A checks and homogenization


def remove_rows_with_missing_time(df: pd.DataFrame, logger=logger, verbose: bool = False):
    """Remove dataframe rows where the ``"time"`` is ``NaT``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    verbose : bool
        Whether to verbose the processing. The default is ``False``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with valid timesteps.
    """
    # Get the number of rows of the dataframe
    n_rows = len(df)
    # Drop rows with "time" nat values
    df = df.dropna(subset="time", axis=0)
    # If no valid timesteps, raise error
    if len(df.index) == 0:
        msg = "There are not valid timestep."
        raise ValueError(msg)
    # Otherwise, report the number of invalid timesteps
    n_invalid_timesteps = n_rows - len(df)
    if n_invalid_timesteps > 0:
        msg = f"{n_invalid_timesteps} rows had invalid timesteps and were discarded."
        log_warning(logger=logger, msg=msg, verbose=verbose)
    return df


def remove_duplicated_timesteps(df: pd.DataFrame, logger=None, verbose: bool = False):
    """Remove duplicated timesteps.

    It keep only the first timestep occurrence !

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    verbose : bool
        Whether to verbose the processing. The default is ``False``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with valid unique timesteps.
    """
    values, counts = np.unique(df["time"], return_counts=True)
    idx_duplicates = np.where(counts > 1)[0]
    values_duplicates = values[idx_duplicates].astype("M8[s]")
    # If there are duplicated timesteps
    if len(values_duplicates) > 0:
        # TODO: raise error if duplicated timesteps have different values !

        # Drop duplicated timesteps (keeping the first occurrence)
        df = df.drop_duplicates(subset="time", keep="first")
        # Report the values of duplicated timesteps
        msg = (
            f"The following timesteps occurred more than once: {values_duplicates}. Only the first occurrence"
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
        raise ValueError(msg)
    return df


def drop_time_periods(df, time_periods):
    """Drop problematic time periods."""
    for time_period in time_periods:
        if len(df) > 0:
            start_time = time_period[0]
            end_time = time_period[1]
            df = df[(df["time"] < start_time) | (df["time"] > end_time)]
    # Check there are row left
    if len(df) == 0:
        msg = "No rows left after removing problematic time_periods. Maybe you need to adjust the issue YAML file."
        raise ValueError(msg)

    return df


def remove_issue_timesteps(df, issue_dict, logger=None, verbose=False):
    """Drop dataframe rows with timesteps listed in the issue dictionary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    issue_dict : dict
        Issue dictionary.
    verbose : bool
        Whether to verbose the processing. The default is ``False``.

    Returns
    -------
    pandas.DataFrame
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

    # Report number of dropped rows
    n_rows_dropped = n_initial_rows - len(df)
    if n_rows_dropped > 0:
        msg = f"{n_rows_dropped} rows were dropped following the issue YAML file content."
        log_info(logger=logger, msg=msg, verbose=verbose)

    return df


def cast_column_dtypes(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """Convert ``'object'`` dataframe columns into DISDRODB L0A dtype standards.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    pandas.DataFrame
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
    df = df.copy(deep=True)  # avoid modify also dtype of input df
    for column in columns:
        try:
            df[column] = df[column].astype(dtype_dict[column])
        except ValueError as e:
            msg = f"ValueError: The column {column} has {e}"
            raise ValueError(msg)
    return df


def coerce_corrupted_values_to_nan(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """Coerce corrupted values in dataframe numeric columns to ``np.nan``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    pandas.DataFrame
        Dataframe with string columns without corrupted values.
    """
    # Cast dataframe to dtypes
    dtype_dict = get_l0a_dtype(sensor_name)

    # Get numeric columns
    numeric_columns = [k for k, dtype in dtype_dict.items() if "float" in dtype or "int" in dtype]

    # Get dataframe column names
    columns = list(df.columns)

    # Cast dataframe columns
    for column in columns:
        if column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def strip_string_spaces(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """Strip leading/trailing spaces from dataframe string columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    pandas.DataFrame
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
                msg = f"The column {column} is not a string/object dtype."
                raise AttributeError(msg)
    return df


def strip_delimiter(string):
    """Remove the first and last delimiter occurrence from a string."""
    if not isinstance(string, str):
        return string
    split_str = infer_split_str(string=string)
    string = string.strip(split_str)
    return string


def strip_delimiter_from_raw_arrays(df):
    """Remove the first and last delimiter occurrence from the raw array fields."""
    # Possible fields
    possible_fields = [
        "raw_drop_number",
        "raw_drop_concentration",
        "raw_drop_average_velocity",
    ]
    available_fields = list(df.columns[np.isin(df.columns, possible_fields)])
    # Loop over the fields and strip away the delimiter
    for field in available_fields:
        df[field] = df[field].apply(strip_delimiter)
    # Return the dataframe
    return df


def is_raw_array_string_not_corrupted(string):
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
            df = df[df[field].apply(is_raw_array_string_not_corrupted)]
    # Check if there are rows left
    if len(df) == 0:
        raise ValueError("No remaining rows after data corruption checks.")
    # If only one row available, raise also error
    if len(df) == 1:
        raise ValueError("Only 1 row remains after data corruption checks. Check the raw file and maybe delete it.")
    # Return the dataframe
    return df


def replace_nan_flags(df, sensor_name, logger=None, verbose=False):
    """Set values corresponding to ``nan_flags`` to ``np.nan``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Whether to verbose the processing. The default is ``False``.

    Returns
    -------
    pandas.DataFrame
        Dataframe without nan_flags values.
    """
    # Get dictionary of nan flags
    dict_nan_flags = get_nan_flags_dict(sensor_name)
    # Loop over the needed variable, and replace nan_flags values with np.nan
    for var, nan_flags in dict_nan_flags.items():
        # If the variable is in the dataframe
        if var in df:
            # Get array with occurrence of nan_flags
            is_a_nan_flag = df[var].isin(nan_flags)
            # If nan_flags values are present, replace with np.nan
            n_nan_flags_values = np.sum(is_a_nan_flag)
            if n_nan_flags_values > 0:
                msg = f"In variable {var}, {n_nan_flags_values} values were nan_flags and were replaced to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                df.loc[is_a_nan_flag, var] = np.nan
    # Return dataframe
    return df


def set_nan_outside_data_range(df, sensor_name, logger=None, verbose=False):
    """Set values outside the data range as ``np.nan``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Whether to verbose the processing. The default is ``False``.

    Returns
    -------
    pandas.DataFrame
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
            n_invalid = np.sum(~is_valid)
            if n_invalid > 0:
                msg = f"{n_invalid} {var} values were outside the data range and were set to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                df[var] = df[var].where(is_valid)  # set not valid to np.nan

    # Return dataframe
    return df


def set_nan_invalid_values(df, sensor_name, logger=None, verbose=False):
    """Set invalid (class) values to ``np.nan``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Whether to verbose the processing. The default is ``False``.

    Returns
    -------
    pandas.DataFrame
        Dataframe without invalid values.
    """
    # Get dictionary of valid values
    dict_valid_values = get_valid_values_dict(sensor_name)
    # Loop over the variable with a defined data_range
    for var, valid_values in dict_valid_values.items():
        # If the variable is in the dataframe
        if var in df:
            # Get array with occurrence of correct values (or already np.nan)
            is_valid_values = df[var].isin(valid_values) | df[var].isna()
            # If invalid values are present, replace with np.nan
            n_invalid_values = np.sum(~is_valid_values)
            if n_invalid_values > 0:
                msg = f"{n_invalid_values} {var} values were invalid and were replaced to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                df[var] = df[var].where(is_valid_values)  # set not valid to np.nan

    # Return dataframe
    return df


def sanitize_df(
    df,
    sensor_name,
    verbose=True,
    issue_dict=None,
    logger=None,
):
    """Read and parse a raw text files into a L0A dataframe.

    Parameters
    ----------
    filepath : str
        File path
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Whether to verbose the processing. The default is ``True``.
    issue_dict : dict
        Issue dictionary providing information on timesteps to remove.
        The default is an empty dictionary ``{}``.
        Valid issue_dict key are ``'timesteps'`` and ``'time_periods'``.
        Valid issue_dict values are list of datetime64 values (with second accuracy).
        To correctly format and check the validity of the ``issue_dict``, use
        the ``disdrodb.l0.issue.check_issue_dict`` function.

    Returns
    -------
    pandas.DataFrame
        Dataframe
    """
    # Define the issue dictionary
    # - If None, set to empty dictionary
    issue_dict = {} if issue_dict is None else issue_dict

    # - Remove rows with time NaT
    df = remove_rows_with_missing_time(df, logger=logger, verbose=verbose)

    # - Remove duplicated timesteps
    df = remove_duplicated_timesteps(df, logger=logger, verbose=verbose)

    # - Filter out problematic tiemsteps reported in the issue YAML file
    df = remove_issue_timesteps(df, issue_dict=issue_dict, logger=logger, verbose=verbose)

    # - Coerce numeric columns corrupted values to np.nan
    df = coerce_corrupted_values_to_nan(df, sensor_name=sensor_name)

    # - Strip trailing/leading space from string columns
    df = strip_string_spaces(df, sensor_name=sensor_name)

    # - Strip first and last delimiter from the raw arrays
    df = strip_delimiter_from_raw_arrays(df)

    # - Remove corrupted rows
    df = remove_corrupted_rows(df)

    # - Cast dataframe to dtypes
    df = cast_column_dtypes(df, sensor_name=sensor_name)

    # - Replace nan flags values with np.nans
    df = replace_nan_flags(df, sensor_name=sensor_name, logger=logger, verbose=verbose)

    # - Set values outside the data range to np.nan
    df = set_nan_outside_data_range(df, sensor_name=sensor_name, logger=logger, verbose=verbose)

    # - Replace invalid values with np.nan
    df = set_nan_invalid_values(df, sensor_name=sensor_name, logger=logger, verbose=verbose)

    # - Sort by time
    df = df.sort_values("time")

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
    filepath: str,
    force: bool = False,
    logger=None,
    verbose: bool = False,
):
    """Save the dataframe into an Apache Parquet file.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    filepath : str
        Output file path.
    force : bool, optional
        Whether to overwrite existing data.
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories. This is the default.
    verbose : bool, optional
        Whether to verbose the processing. The default is ``False``.

    Raises
    ------
    ValueError
        The input dataframe can not be written as an Apache Parquet file.
    NotImplementedError
        The input dataframe can not be processed.
    """
    # -------------------------------------------------------------------------.
    # Create station directory if does not exist
    create_directory(os.path.dirname(filepath))

    # Check if the file already exists
    # - If force=True --> Remove it
    # - If force=False --> Raise error
    remove_if_exists(filepath, force=force, logger=logger)

    # -------------------------------------------------------------------------.
    # Define writing options
    compression = "snappy"  # 'gzip', 'brotli, 'lz4', 'zstd'
    row_group_size = 100000
    engine = "pyarrow"
    # -------------------------------------------------------------------------.
    # Save dataframe to Apache Parquet
    try:
        df.to_parquet(
            filepath,
            engine=engine,
            compression=compression,
            row_group_size=row_group_size,
        )
        msg = f"The Pandas Dataframe has been written as an Apache Parquet file to {filepath}."
        log_info(logger=logger, msg=msg, verbose=verbose)
    except Exception as e:
        msg = f"The Pandas DataFrame cannot be written as an Apache Parquet file. The error is: \n {e}."
        raise ValueError(msg)
    # -------------------------------------------------------------------------.


####--------------------------------------------------------------------------.
#### DISDRODB L0A product reader


def concatenate_dataframe(list_df: list, logger=None, verbose: bool = False) -> pd.DataFrame:
    """Concatenate a list of dataframes.

    Parameters
    ----------
    list_df : list
        List of dataframes.
    verbose : bool, optional
        If ``True``, print messages.
        If ``False``, no print.

    Returns
    -------
    pandas.DataFrame
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
    msg = "Concatenation of dataframes started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # Concatenate the dataframe
    try:
        df = pd.concat(list_df, axis=0, ignore_index=True)
        # Sort by increasing time
        df = df.sort_values(by="time")

    except (AttributeError, TypeError) as e:
        msg = f"Can not concatenate the files. \n Error: {e}"
        raise ValueError(msg)

    # Log
    msg = "Concatenation of dataframes has finished."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # Return dataframe
    return df


def _read_l0a(filepath: str, verbose: bool = False, logger=None, debugging_mode: bool = False) -> pd.DataFrame:
    # Log
    msg = f"Reading L0 Apache Parquet file at {filepath} started."
    log_info(logger=logger, msg=msg, verbose=verbose)
    # Open file
    df = pd.read_parquet(filepath)
    if debugging_mode:
        df = df.iloc[0:100]
    # Log
    msg = f"Reading L0 Apache Parquet file at {filepath} ended."
    log_info(logger=logger, msg=msg, verbose=verbose)
    return df


def read_l0a_dataframe(
    filepaths: Union[str, list],
    verbose: bool = False,
    logger=None,
    debugging_mode: bool = False,
) -> pd.DataFrame:
    """Read DISDRODB L0A Apache Parquet file(s).

    Parameters
    ----------
    filepaths : str or list
        Either a list or a single filepath.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is ``False``.
    debugging_mode : bool
        If ``True``, it reduces the amount of data to process.
        If filepaths is a list, it reads only the first 3 files.
        For each file it select only the first 100 rows.
        The default is ``False``.

    Returns
    -------
    pandas.DataFrame
        L0A Dataframe.

    """
    from disdrodb.l0.l0a_processing import concatenate_dataframe

    # ----------------------------------------
    # Check filepaths validity
    if not isinstance(filepaths, (list, str)):
        raise TypeError("Expecting filepaths to be a string or a list of strings.")

    # ----------------------------------------
    # If filepath is a string, convert to list
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    # ---------------------------------------------------
    # If debugging_mode=True, it reads only the first 3 filepaths
    if debugging_mode:
        filepaths = filepaths[0:3]  # select first 3 filepaths

    # ---------------------------------------------------
    # Define the list of dataframe
    list_df = [
        _read_l0a(filepath, verbose=verbose, logger=logger, debugging_mode=debugging_mode) for filepath in filepaths
    ]

    # Concatenate dataframe
    df = concatenate_dataframe(list_df, logger=logger, verbose=verbose)

    # Ensure time is in nanoseconds
    df["time"] = df["time"].astype("M8[ns]")

    # ---------------------------------------------------
    # Return dataframe
    return df


####---------------------------------------------------------------------------.
#### L0A Utility


def read_raw_text_files(
    filepaths: Union[list, str],
    reader,
    sensor_name,
    verbose=True,
    logger=None,
) -> pd.DataFrame:
    """Read and parse a list for raw files into a dataframe.

    Parameters
    ----------
    filepaths : Union[list,str]
        File(s) path(s)
    reader:
        DISDRODB reader function.
        Format: reader(filepath, logger=None)
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Whether to verbose the processing. The default is ``True``.

    Returns
    -------
    pandas.DataFrame
        Dataframe

    Raises
    ------
    ValueError
        Input parameters can not be used or the raw file can not be processed.

    """
    # ------------------------------------------------------.
    # Check input list
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    if len(filepaths) == 0:
        raise ValueError("'filepaths' must contains at least 1 filepath.")

    # ------------------------------------------------------.
    # Loop over all raw files
    n_files = len(filepaths)
    processed_file_counter = 0
    list_skipped_files_msg = []
    list_df = []
    for filepath in filepaths:
        # Try read the raw text file
        try:
            df = reader(filepath, logger=logger)
            # Sanitize the dataframe
            df = sanitize_df(
                df=df,
                sensor_name=sensor_name,
                logger=logger,
                verbose=verbose,
            )
            # Append dataframe to the list
            list_df.append(df)
            # Update the logger
            processed_file_counter += 1
            msg = f"Raw file '{filepath}' processed successfully ({processed_file_counter}/{n_files})."
            log_info(logger=logger, msg=msg, verbose=verbose)

        # Skip the file if the processing fails
        except Exception as e:
            # Update the logger
            msg = f"{filepath} has been skipped. The error is: {e}."
            log_error(logger=logger, msg=msg, verbose=verbose)
            list_skipped_files_msg.append(msg)

    # Update logger
    msg = f"{len(list_skipped_files_msg)} of {n_files} have been skipped."
    log_info(logger=logger, msg=msg, verbose=verbose)

    ##----------------------------------------------------------------.
    # Concatenate the dataframe
    if len(list_df) == 0:
        raise ValueError("Any raw file could be read!")
    df = concatenate_dataframe(list_df, verbose=verbose, logger=logger)

    # ------------------------------------------------------.
    # Enforce output time to be [ns]
    # --> For compatibility with xarray
    df["time"] = df["time"].astype("M8[ns]")

    # Return the dataframe
    return df
