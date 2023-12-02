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
"""Define DISDRODB Data Input/Output."""
import logging
import os
from typing import Union

import pandas as pd

from disdrodb.api.path import define_l0a_station_dir
from disdrodb.utils.directories import list_files
from disdrodb.utils.logger import log_info

logger = logging.getLogger(__name__)


####--------------------------------------------------------------------------.
#### List Station Files


def _check_glob_pattern(pattern: str) -> None:
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


def _check_glob_patterns(patterns: Union[str, list]) -> list:
    """Check if glob patterns are valids."""
    if not isinstance(patterns, (str, list)):
        raise ValueError("'glob_patterns' must be a str or list of strings.")
    if isinstance(patterns, str):
        patterns = [patterns]
    _ = [_check_glob_pattern(pattern) for pattern in patterns]
    return patterns


def _get_file_list(raw_dir: str, station_name, glob_pattern) -> list:
    """Get the list of files from a directory based on glob pattern.

    Parameters
    ----------
    raw_dir : str
        Campaign directory of the raw data.
    station_name: str
        Name of the station.
    glob_pattern : str
        Pattern to match.

    Returns
    -------
    list
        List of file paths.
    """
    data_dir = os.path.join(raw_dir, "data", station_name)
    filepaths = list_files(data_dir, glob_pattern=glob_pattern, recursive=True)
    filepaths = sorted(filepaths)
    return filepaths


def _get_available_filepaths(raw_dir, station_name, glob_patterns):
    # Retrieve filepaths list
    filepaths = [_get_file_list(raw_dir, station_name, glob_pattern=pattern) for pattern in glob_patterns]
    filepaths = [x for xs in filepaths for x in xs]  # flatten list

    # Check there are files
    n_files = len(filepaths)
    if n_files == 0:
        glob_filepath_patterns = [os.path.join(raw_dir, pattern) for pattern in glob_patterns]
        raise ValueError(f"No file found at {glob_filepath_patterns}.")
    return filepaths


def _filter_filepaths(filepaths, debugging_mode):
    """Filter out filepaths if debugging_mode=True."""
    if debugging_mode:
        max_files = min(3, len(filepaths))
        filepaths = filepaths[0:max_files]
    return filepaths


def get_raw_filepaths(raw_dir, station_name, glob_patterns, verbose=False, debugging_mode=False):
    """Get the list of files from a directory based on input parameters.

    Currently concatenates all files provided by the glob patterns.
    In future, this might be modified to enable DISDRODB processing when raw data
    are separated in multiple files.

    Parameters
    ----------
    raw_dir : str
        Directory of the campaign where to search for files.
        Format <..>/DISDRODB/Raw/<DATA_SOURCE>/<CAMPAIGN_NAME>
    station_name : str
        ID of the station
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.
    debugging_mode : bool, optional
        If True, it select maximum 3 files for debugging purposes.
        The default is False.

    Returns
    -------
    filepaths : list
        List of files file paths.

    """
    glob_patterns = _check_glob_patterns(glob_patterns)

    filepaths = _get_available_filepaths(raw_dir=raw_dir, station_name=station_name, glob_patterns=glob_patterns)

    # Filter out filepaths if debugging_mode=True
    filepaths = _filter_filepaths(filepaths, debugging_mode)

    # Log number of files to process
    n_files = len(filepaths)
    data_dir = os.path.join(raw_dir, "data", station_name)
    msg = f" - {n_files} files to process in {data_dir}"
    log_info(logger=logger, msg=msg, verbose=verbose)

    # Return file list
    return filepaths


def get_l0a_filepaths(processed_dir, station_name, debugging_mode=False):
    """Retrieve L0A files for a give station.

    Parameters
    ----------
    processed_dir : str
        Directory of the campaign where to search for the L0A files.
        Format <..>/DISDRODB/Processed/<DATA_SOURCE>/<CAMPAIGN_NAME>
    station_name : str
        ID of the station
    debugging_mode : bool, optional
        If True, it select maximum 3 files for debugging purposes.
        The default is False.

    Returns
    -------
    filepaths : list
        List of L0A file paths.

    """
    station_dir = define_l0a_station_dir(processed_dir, station_name)
    filepaths = list_files(station_dir, glob_pattern="*.parquet", recursive=True)

    # Filter out filepaths if debugging_mode=True
    filepaths = _filter_filepaths(filepaths, debugging_mode=debugging_mode)

    # If no file available, raise error
    if len(filepaths) == 0:
        msg = f"No L0A Apache Parquet file is available in {station_dir}. Run L0A processing first."
        raise ValueError(msg)

    return filepaths


####--------------------------------------------------------------------------.
#### DISDRODB L0A product reader


def _read_l0a(filepath: str, verbose: bool = False, debugging_mode: bool = False) -> pd.DataFrame:
    # Log
    msg = f" - Reading L0 Apache Parquet file at {filepath} started."
    log_info(logger, msg, verbose)
    # Open file
    df = pd.read_parquet(filepath)
    if debugging_mode:
        df = df.iloc[0:100]
    # Log
    msg = f" - Reading L0 Apache Parquet file at {filepath} ended."
    log_info(logger, msg, verbose)
    return df


def read_l0a_dataframe(
    filepaths: Union[str, list],
    verbose: bool = False,
    debugging_mode: bool = False,
) -> pd.DataFrame:
    """Read DISDRODB L0A Apache Parquet file(s).

    Parameters
    ----------
    filepaths : str or list
        Either a list or a single filepath .
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        If filepaths is a list, it reads only the first 3 files
        For each file it select only the first 100 rows.
        The default is False.

    Returns
    -------
    pd.DataFrame
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
    # - If debugging_mode=True, it reads only the first 3 filepaths
    if debugging_mode:
        filepaths = filepaths[0:3]  # select first 3 filepaths

    # - Define the list of dataframe
    list_df = [_read_l0a(filepath, verbose=verbose, debugging_mode=debugging_mode) for filepath in filepaths]
    # - Concatenate dataframe
    df = concatenate_dataframe(list_df, verbose=verbose)
    # ---------------------------------------------------
    # Return dataframe
    return df
