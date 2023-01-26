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
import glob
from xmlrpc.server import list_public_methods
import pandas as pd
import dask.dataframe as dd
import logging
import tarfile
from typing import Union

from disdrodb.L0.standards import get_L0A_dtype
from disdrodb.L0.check_standards import check_L0A_standards, check_L0A_column_names
from disdrodb.L0.io import _remove_if_exists
from disdrodb.L0.io import infer_station_id_from_fpath
from disdrodb.utils.logger import log_info, log_warning, log_debug

logger = logging.getLogger(__name__)

# TODO:
# Renaming:
# - read_L0A_raw_file_list --> read_raw_file_list ?
# - write_df_to_parquet --> write_L0A?
# Remove or refactor
# - read_raw_data_zipped


def check_glob_pattern(pattern: str) -> None:
    """Check if the input parameters is a string and if it can be used as pattern.

    Parameters
    ----------
    pattern : str
        String to be checked.

    Raises
    ------
    TypeError
        The input parameter is not a string.
    ValueError
        The input parameter can not be used as pattern.
    """
    if not isinstance(pattern, str):
        raise TypeError("Expect pattern as a string.")
    if pattern[0] == "/":
        raise ValueError("glob_pattern should not start with /")


def _get_file_list(raw_dir: str, glob_pattern) -> list:
    """Get the list of files from a directory based on pattern.

    Parameters
    ----------
    raw_dir : _type_
        Directory of the raw dataset.
    glob_pattern : _type_
        Pattern to match

    Returns
    -------
    list
        List of file paths.
    """
    check_glob_pattern(glob_pattern)
    glob_fpath_pattern = os.path.join(raw_dir, glob_pattern)
    list_fpaths = sorted(glob.glob(glob_fpath_pattern))
    return list_fpaths


def get_file_list(raw_dir, glob_pattern, verbose=False, debugging_mode=False):
    """Get the list of files from a directory based on input parameters.

    Parameters
    ----------
    raw_dir : str
        Directory of the campaign where to search for files.
    glob_pattern : str or list
        Glob pattern to search for files. Can also be a list of glob patterns.
    verbose : bool, optional
        Wheter to verbose the processing.
        The default is False.
    debugging_mode : bool, optional
        If True, it select maximum 3 files for debugging purposes.
        The default is False.

    Returns
    -------
    list_fpaths : list
        List of files file paths.

    """
    if not isinstance(glob_pattern, (str, list)):
        raise ValueError("'glob_pattern' must be a str or list of strings.")
    if isinstance(glob_pattern, str):
        glob_pattern = [glob_pattern]

    # Retrieve filepaths list
    list_fpaths = [_get_file_list(raw_dir, pattern) for pattern in glob_pattern]
    list_fpaths = [x for xs in list_fpaths for x in xs]  # flatten list

    # Check there are files
    n_files = len(list_fpaths)
    if n_files == 0:
        glob_fpath_patterns = [
            os.path.join(raw_dir, pattern) for pattern in glob_pattern
        ]
        raise ValueError(f"No file found at t {glob_fpath_patterns}.")

    # Subset file_list if debugging_mode
    if debugging_mode:
        max_files = min(3, n_files)
        list_fpaths = list_fpaths[0:max_files]

    # Log
    n_files = len(list_fpaths)
    msg = f" - {n_files} files to process in {raw_dir}"
    if verbose:
        print(msg)
    logger.info(msg)

    # Return file list
    return list_fpaths


####---------------------------------------------------------------------------.
#### Dataframe creation
def preprocess_reader_kwargs(reader_kwargs: dict, lazy: bool = True) -> dict:
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

    # TODO: Remove this when removing read_raw_data_zipped
    if reader_kwargs.get("zipped", False):
        reader_kwargs.pop("zipped", None)
        reader_kwargs.pop("blocksize", None)
        reader_kwargs.pop("file_name_to_read_zipped", None)

    return reader_kwargs


def concatenate_dataframe(
    list_df: list, verbose: bool = False, lazy: bool = True
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Concatenate a list of dataframes.

    Parameters
    ----------
    list_df : list
        List of dataframes.
    verbose : bool, optional
        If True, print messages.
        If False, no print.
    lazy : bool, optional
        If True : Dask is used.
        If False : Pandas is used.

    Returns
    -------
    Union[pd.DataFrame,dd.DataFrame]
        Concatenated dataframe.

    Raises
    ------
    ValueError
        Concatenation can not be done.
    """
    # Import dask or pandas
    if lazy:
        import dask.dataframe as dd
    else:
        import pandas as dd
    # Log
    msg = " - Concatenation of dataframes started."
    log_info(logger, msg, verbose)
    # Concatenate the dataframe
    try:
        df = dd.concat(list_df, axis=0, ignore_index=True)
        # Drop duplicated values
        df = df.drop_duplicates(subset="time")
        # Sort by increasing time
        df = df.sort_values(by="time")

    except (AttributeError, TypeError) as e:
        msg = f" - Can not create concat data files. \n Error: {e}"
        logger.error(msg)
        raise ValueError(msg)
    # Log
    msg = " - Concatenation of dataframes has finished."
    log_info(logger, msg, verbose)
    # Return dataframe
    return df


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

    # Cast dataframe columns (TODO: tmp fix)
    if isinstance(df, dd.DataFrame):
        # Cast dataframe columns
        for column in columns:
            if column in numeric_columns:
                try:
                    df[column] = dd.to_numeric(df[column], errors="coerce")
                except AttributeError:
                    raise (
                        f"AttributeError: The column {column} is not a numeric column."
                    )
    else:

        for column in columns:
            if column in numeric_columns:
                try:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                except AttributeError:
                    raise (
                        f"AttributeError: The column {column} is not a numeric column."
                    )

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


def read_raw_data(
    filepath: str, column_names: list, reader_kwargs: dict, lazy: bool = True
) -> pd.DataFrame:
    """Read raw data into a dataframe

    Parameters
    ----------
    filepath : str
        Raw file path.
    column_names : list
        Column names.
    reader_kwargs : dict
        Dask or Pandas reading parameters
    lazy : bool, optional
        If True : Dask is used.
        If False : Pandas is used.

    Returns
    -------
    pandas.DataFrame or dask.DataFrame
        Pandas or dask dataframe
    """
    # Preprocess reader_kwargs
    reader_kwargs = preprocess_reader_kwargs(reader_kwargs, lazy=lazy)

    # Enforce all raw files columns with dtype = 'object'
    dtype = "object"

    # Read with pandas or dask
    # - Dask
    if lazy:
        try:
            df = dd.read_csv(filepath, names=column_names, dtype=dtype, **reader_kwargs)
        except dd.errors.EmptyDataError:
            msg = f" - Is empty, skip file: {filepath}"
            logger.exception(msg)
            print(msg)
            pass
    # Pandas
    else:
        reader_kwargs.pop("blocksize", None)
        try:
            df = pd.read_csv(filepath, names=column_names, dtype=dtype, **reader_kwargs)
        except pd.errors.EmptyDataError:
            msg = f" - Is empty, skip file: {filepath}"
            logger.exception(msg)
            print(msg)
            pass
    # Return dataframe
    return df


def read_raw_data_zipped(
    filepath: str, column_names: list, reader_kwargs: dict, lazy: bool = True
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
    lazy : bool, optional
        If True : Dask is used.
        If False : Pandas is used.

    Returns
    -------
    pandas.DataFrame or dask.DataFrame
        Pandas or dask dataframe

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
                    lazy=lazy,
                )
                break
            else:
                # Read the data
                df_temp = read_raw_data(
                    filepath=tar.extractfile(file),
                    column_names=column_names,
                    reader_kwargs=reader_kwargs,
                    lazy=lazy,
                )

                # Concat all files in tar
                df = df.append(df_temp)

        except pd.errors.EmptyDataError:
            msg = f" - Is empty, skip file: {file}"
            logger.exception(msg)
            raise pd.errors.EmptyDataError(msg)
            pass
        except pd.errors.ParserError:
            msg = f" - Cannot parse, skip file: {file}"
            logger.exception(msg)
            raise pd.errors.ParserError(msg)
            pass
        except UnicodeDecodeError:
            msg = f" - Unicode error, skip file: {file}"
            logger.exception(msg)
            raise UnicodeDecodeError(msg)
            pass

    # Close zipped file
    tar.close()

    return df


def read_L0A_raw_file_list(
    file_list: Union[list, str],
    column_names: list,
    reader_kwargs: dict,
    sensor_name: str,
    verbose: bool,
    df_sanitizer_fun: object = None,
    lazy: bool = False,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Read and parse a list for raw files into a dataframe.

    Parameters
    ----------
    file_list : Union[list,str]
        File(s) path(s)
    column_names : list
        Columns names.
    reader_kwargs : dict
        Dask or Pandas reading parameters.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Wheter to verbose the processing.
    df_sanitizer_fun : object, optional
        Sanitizer function to format the datafame.
    lazy : bool, optional
        If True : Dask is used.
        If False : Pandas is used.

    Returns
    -------
    Union[pd.DataFrame,dd.DataFrame]
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
        # TODO check df_sanitizer_fun has only lazy and df arguments !

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
        # Try to process a raw file
        try:
            # -----------------------------------------------------------------.
            # ---------------> THIS SHOULD BE REFACTORED  <-------------------.
            # Open the zip and choose the raw file (for GPM campaign)
            if reader_kwargs.get("zipped"):
                df = read_raw_data_zipped(
                    filepath=filepath,
                    column_names=column_names,
                    reader_kwargs=reader_kwargs,
                    lazy=lazy,
                )

            else:
                # Read the data
                df = read_raw_data(
                    filepath=filepath,
                    column_names=column_names,
                    reader_kwargs=reader_kwargs,
                    lazy=lazy,
                )

            # Check if file empty
            if len(df.index) == 0:
                msg = f" - {filepath} is empty and has been skipped."
                log_warning(logger, msg, verbose)
                list_skipped_files_msg.append(msg)
                continue

            # Check column number, ignore if columns_names empty
            if len(column_names) != 0:
                if len(df.columns) != len(column_names):
                    msg = (
                        f" - {filepath} has wrong columns number, and has been skipped."
                    )
                    log_warning(logger, msg, verbose)
                    list_skipped_files_msg.append(msg)
                    continue
            # -------------------> TILL HERE   <------------------------------.
            # ----------------------------------------------------------------.
            # Sanitize the dataframe with a custom function
            if df_sanitizer_fun is not None:
                df = df_sanitizer_fun(df, lazy=lazy)

            # Remove duplicated timesteps
            # - TODO: Log info !!!
            df = df.drop_duplicates(subset="time", keep="first")

            # ------------------------------------------------------.
            # Check column names met DISDRODB standards
            check_L0A_column_names(df, sensor_name=sensor_name)

            # ------------------------------------------------------.
            # Append dataframe to the list
            list_df.append(df)

            # Update the logger
            processed_file_counter += 1
            logger.debug(
                f"{processed_file_counter} / {n_files} processed successfully. File name: {filepath}"
            )

        # If processing of raw file fails
        except (Exception, ValueError) as e:
            # Update the logger
            msg = f" - {filepath} has been skipped. \n -- The error is: {e}."
            log_warning(logger, msg, verbose)
            list_skipped_files_msg.append(msg)

    # Update logger
    msg = f" - {len(list_skipped_files_msg)} of {n_files} have been skipped."
    log_info(logger, msg, verbose)
    logger.info("---")
    logger.info(msg)
    logger.info("---")

    ##----------------------------------------------------------------.
    #### - Concatenate the dataframe
    if len(list_df) == 0:
        raise ValueError(f"No dataframe to return. Impossible to parse {file_list}.")
    if len(list_df) > 1:
        df = concatenate_dataframe(list_df, verbose=verbose, lazy=lazy)
    else:
        df = list_df[0]

    # ------------------------------------------------------.
    # Final dataframe cleaning
    # - Coerce numeric columns corrupted values to np.nan
    df = coerce_corrupted_values_to_nan(df, sensor_name=sensor_name)

    # - Strip trailing/leading space from string columns
    df = strip_string_spaces(df, sensor_name=sensor_name)

    # - Remove rows with duplicate timestep (keep the first)
    df = df.drop_duplicates(subset=["time"])

    # - Cast dataframe to dtypes
    df = cast_column_dtypes(df, sensor_name=sensor_name)

    # ------------------------------------------------------.
    #### - Filter out problematic data reported in issue file
    # TODO: [TEST IMPLEMENTATION] remove_problematic_timestamp in dev/TODO_issue_code.py
    # issue_dict = read_issue(raw_dir, station_id)
    # df = remove_problematic_timestamp(df, issue_dict, verbose)

    # ------------------------------------------------------.
    # Return the dataframe
    return df


####---------------------------------------------------------------------------.
#### Parquet Writer
def _write_to_parquet(
    df: Union[pd.DataFrame, dd.DataFrame], fpath: str, force: bool = False
):

    import pandas as pd
    import dask.dataframe
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
    # Save to parquet
    # - If Pandas df
    if isinstance(df, pd.DataFrame):
        try:
            df.to_parquet(
                fpath,
                engine=engine,
                compression=compression,
                row_group_size=row_group_size,
            )
            logger.info(
                f"The Pandas Dataframe has been written as an Apache Parquet file to {fpath}."
            )
        except (Exception) as e:
            msg = f" - The Pandas DataFrame cannot be written as an Apache Parquet file. The error is: \n {e}."
            logger.exception(msg)
            raise ValueError(msg)

    # - If Dask DataFrame
    elif isinstance(df, dask.dataframe.DataFrame):
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html
        try:
            # df.repartition(npartitions=1)
            _ = df.to_parquet(
                fpath,
                schema="infer",
                engine=engine,
                row_group_size=row_group_size,
                compression=compression,
                write_metadata_file=False,
            )
            logger.info(
                f"The Dask Dataframe has been written as an Apache Parquet file to {fpath}."
            )
        except (Exception) as e:
            msg = f" - The Dask DataFrame cannot be written as an Apache Parquet file. The error is: \n {e}."
            logger.exception(msg)
            raise ValueError(msg)
    else:
        raise NotImplementedError("Pandas or Dask DataFrame is required.")

    # -------------------------------------------------------------------------.


def write_df_to_parquet(
    df: Union[pd.DataFrame, dd.DataFrame],
    fpath: str,
    force: bool = False,
    verbose: bool = False,
):
    """Save the dataframe into an Apache Parquet file.

    Parameters
    ----------
    df : Union[pd.DataFrame,dd.DataFrame]
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
