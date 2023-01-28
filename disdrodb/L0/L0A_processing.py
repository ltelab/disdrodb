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

# File content
# - Functions to process raw data files into L0A Apache parquet.

# -----------------------------------------------------------------------------.
import os
import inspect
import logging
import tarfile
import pandas as pd
import numpy as np
from typing import Union
from disdrodb.L0.standards import get_L0A_dtype
from disdrodb.L0.check_standards import check_L0A_column_names
from disdrodb.L0.io import _remove_if_exists
from disdrodb.L0.io import infer_station_id_from_fpath

# Logger
from disdrodb.utils.logger import (
    log_info,
    log_warning,
    log_error,
    log_debug,
)

logger = logging.getLogger(__name__)

# Possible renaming:
# - write_df_to_parquet --> write_L0A?
# Remove or refactor
# - _read_raw_data_zipped

####---------------------------------------------------------------------------.
#### Raw file readers


def preprocess_reader_kwargs(reader_kwargs: dict, lazy: bool = False) -> dict:
    """Define a dictionary with the parameters required for reading the raw data with Pandas or Dask.

    Parameters
    ----------
    reader_kwargs : dict
        Initial parameter dictionary.
    lazy : bool, optional
        If True : Dask is used.
        If False : Pandas is used.
    Returns
    -------
    dict
        Parameter dictionary that matches either Pandas or Dask.
    """
    # Remove dtype key
    # - The dtype is enforced to be 'object' in the read function !
    reader_kwargs.pop("dtype", None)

    # Preprocess the reader_kwargs
    reader_kwargs = reader_kwargs.copy()

    if lazy:
        reader_kwargs.pop("index_col", None)
    if not lazy:
        reader_kwargs.pop("blocksize", None)
        reader_kwargs.pop("assume_missing", None)

    # TODO: Remove this when removing _read_raw_data_zipped
    if reader_kwargs.get("zipped", False):
        reader_kwargs.pop("zipped", None)
        reader_kwargs.pop("blocksize", None)
        reader_kwargs.pop("file_name_to_read_zipped", None)

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

    # Read with pandas
    reader_kwargs.pop("blocksize", None)

    # If zipped in reader_kwargs, use __read_raw_data_zipped
    if reader_kwargs.get("zipped"):
        df = _read_raw_data_zipped(
            filepath=filepath,
            column_names=column_names,
            reader_kwargs=reader_kwargs,
        )
    else:
        try:
            df = pd.read_csv(filepath, names=column_names, dtype=dtype, **reader_kwargs)
        except pd.errors.EmptyDataError:
            msg = f" - Is empty, skip file: {filepath}"
            log_warning(logger=logger, msg=msg, verbose=False)
            pass
    # Return dataframe
    return df


def _read_raw_data_zipped(
    filepath: str,
    column_names: list,
    reader_kwargs: dict,
) -> pd.DataFrame:
    """Read zipped raw data into a dataframe.
    Used because some campaign has tar with multiple files inside,
    and in some situation only one files has to be read.
    Tar reading work only with pandas.
    Put the only file name to read into file_name_to_read_zipped variable,
    if file_name_to_read_zipped is none, all the tar contenet will be
    read and concat into a single dataframe.


    Parameters
    ----------
    filepath : str
        Raw file path.
    column_names : list
        Column names.
    reader_kwargs : dict
        Dask or Pandas reading parameters

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe

    Raises
    ------
    pd.errors.EmptyDataError
        File is empty
    pd.errors.ParserError
        File can not be read
    UnicodeDecodeError
        File can not be decoded
    """

    df = pd.DataFrame()
    tar = tarfile.open(filepath)

    file_name_to_read_zipped = reader_kwargs.get("file_name_to_read_zipped")

    # TODO: deprecate reading zipped files !
    # This function should ready only a single file !

    # Loop tar files
    for file in tar.getnames():
        # Check if pass only particular file to read
        if file_name_to_read_zipped is not None:
            if file.endswith(file_name_to_read_zipped):
                filepath = file
            else:
                continue

        try:
            # If need only to read one file, exit loop file in tar
            if file_name_to_read_zipped is not None:
                # Read the data
                df = read_raw_data(
                    filepath=tar.extractfile(filepath),
                    column_names=column_names,
                    reader_kwargs=reader_kwargs,
                )
                break
            else:
                # Read the data
                df_temp = read_raw_data(
                    filepath=tar.extractfile(file),
                    column_names=column_names,
                    reader_kwargs=reader_kwargs,
                )

                # Concat all files in tar
                df = pd.concat((df, df_temp))

        except pd.errors.EmptyDataError:
            msg = f" - Is empty, skip file: {file}"
            log_warning(logger=logger, msg=msg, verbose=False)
            pass
        except pd.errors.ParserError:
            msg = f" - Cannot parse, skip file: {file}"
            log_error(logger=logger, msg=msg, verbose=False)
            raise pd.errors.ParserError(msg)
        except UnicodeDecodeError:
            msg = f" - Unicode error, skip file: {file}"
            log_error(logger=logger, msg=msg, verbose=False)
            raise UnicodeDecodeError(msg)

    # Close zipped file
    tar.close()

    return df


####---------------------------------------------------------------------------.
#### L0A checks and homogenization


def _check_df_sanitizer_fun(df_sanitizer_fun):
    if not np.all(np.isin(inspect.getfullargspec(df_sanitizer_fun).args, ["df"])):
        raise ValueError(
            "The `df_sanitizer_fun` must have only `df` as input argument!"
        )


def cast_column_dtypes(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """Convert 'object' dataframe columns into DISDRODB L0A dtype standards.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    pd.DataFrame
        Dataframe with corrected columns types.
    """

    # Cast dataframe to dtypes
    dtype_dict = get_L0A_dtype(sensor_name)
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
            raise (f"ValueError: The column {column} has {e}")
    return df


def coerce_corrupted_values_to_nan(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """Coerce corrupted values in dataframe numeric columns to np.nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    pd.DataFrame
        Dataframe with string columns without corrupted values.
    """
    # Cast dataframe to dtypes
    dtype_dict = get_L0A_dtype(sensor_name)

    # Get string columns
    numeric_columns = [
        k for k, dtype in dtype_dict.items() if "float" in dtype or "int" in dtype
    ]

    # Get dataframe column names
    columns = list(df.columns)

    # Cast dataframe columns
    for column in columns:
        if column in numeric_columns:
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            except AttributeError:
                raise ValueError(f"The column {column} is not a numeric column.")
    return df


def strip_string_spaces(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """Strip leading/trailing spaces from dataframe string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    pd.DataFrame
        Dataframe with string columns without leading/trailing spaces.
    """
    # Cast dataframe to dtypes
    dtype_dict = get_L0A_dtype(sensor_name)

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
                raise (
                    f"AttributeError: The column {column} is not a string/object dtype."
                )
    return df


def process_raw_file(
    filepath,
    column_names,
    reader_kwargs,
    df_sanitizer_fun,
    sensor_name,
    verbose,
):
    """Read and parse a raw files into a L0A dataframe.

    Parameters
    ----------
    filepath : str
        File path
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
    """

    # Read the data
    df = read_raw_data(
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
    )

    # Check if file empty
    if len(df.index) == 0:
        msg = f" - {filepath} is empty and has been skipped."
        log_error(logger=logger, msg=msg, verbose=verbose)
        raise ValueError(msg)

    # Check dataframe column number matches columns_names
    n_columns = len(df.columns)
    n_expected_columns = len(column_names)
    if n_columns != n_expected_columns:
        msg = f" - {filepath} has {n_columns} columns, while {n_expected_columns} are expected !."
        log_error(logger, msg, verbose)
        raise ValueError(msg)

    # Sanitize the dataframe with a custom function
    if df_sanitizer_fun is not None:
        # _check_df_sanitizer_fun(df_sanitizer_fun) # TODO activate after refactoring
        df = df_sanitizer_fun(df, lazy=False)

    # Remove rows with time NaT
    # - TODO: Log info !!!
    df = df.dropna(subset="time", axis=0)
    if len(df.index) == 0:
        msg = f" - {filepath} has not valid timestep."
        log_error(logger=logger, msg=msg, verbose=verbose)
        raise ValueError(msg)

    # Remove duplicated timesteps
    # - TODO: Log info !!!
    df = df.drop_duplicates(subset="time", keep="first")

    # ------------------------------------------------------.
    #### - Filter out problematic data reported in issue file
    # TODO: [TEST IMPLEMENTATION] remove_problematic_timestamp in dev/TODO_issue_code.py
    # issue_dict = read_issue(raw_dir, station_id)
    # df = remove_problematic_timestamp(df, issue_dict, verbose)

    # ------------------------------------------------------.
    # - Coerce numeric columns corrupted values to np.nan
    df = coerce_corrupted_values_to_nan(df, sensor_name=sensor_name)

    # - Strip trailing/leading space from string columns
    df = strip_string_spaces(df, sensor_name=sensor_name)

    # - Cast dataframe to dtypes
    df = cast_column_dtypes(df, sensor_name=sensor_name)

    # ------------------------------------------------------.
    # Check column names agrees to DISDRODB standards
    check_L0A_column_names(df, sensor_name=sensor_name)

    # TODO: check_L0A_standards(fpath)
    # ------------------------------------------------------.
    # Return the L0A dataframe
    return df


####---------------------------------------------------------------------------.
#### L0A Apache Parquet Writer
def _write_to_parquet(df: pd.DataFrame, fpath: str, force: bool = False):
    import pandas as pd
    from disdrodb.L0.io import _create_directory

    # -------------------------------------------------------------------------.
    # Check if a file already exists (and remove if force=True)
    _remove_if_exists(fpath, force=force)
    # Cannot create the station folder, so has to be created manually
    _create_directory(os.path.dirname(fpath))

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
    except (Exception) as e:
        msg = f" - The Pandas DataFrame cannot be written as an Apache Parquet file. The error is: \n {e}."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)

    # -------------------------------------------------------------------------.


def write_df_to_parquet(
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

    # Log
    msg = " - Conversion to Apache Parquet started."
    log_info(logger, msg, verbose)
    # Write to Parquet
    _write_to_parquet(df=df, fpath=fpath, force=force)
    # Log
    msg = " - Conversion to Apache Parquet ended."
    log_info(logger, msg, verbose)
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
    # ### Checks arguments
    if df_sanitizer_fun is not None:
        if not callable(df_sanitizer_fun):
            raise ValueError("'df_sanitizer_fun' must be a function.")

    # TODO check df_sanitizer_fun has only df argument !

    if isinstance(file_list, str):
        file_list = [file_list]
    if len(file_list) == 0:
        raise ValueError("'file_list' must contains at least 1 filepath.")

    # ------------------------------------------------------.
    ### - Get station id from file_list
    station_id = infer_station_id_from_fpath(file_list[0])

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
    df = df.drop_duplicates(subset=["time"], axis=0)

    # ------------------------------------------------------.
    # Return the dataframe
    return df
