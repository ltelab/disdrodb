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

from disdrodb.api.io import filter_filepaths
from disdrodb.utils.directories import list_files

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


def get_raw_filepaths(raw_dir, station_name, glob_patterns, verbose=False, logger=logger, debugging_mode=False):
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
        Name of the station.
    verbose : bool, optional
        Whether to verbose the processing.
        The default is ``False``.
    debugging_mode : bool, optional
        If ``True``, it select maximum 3 files for debugging purposes.
        The default is ``False``.

    Returns
    -------
    filepaths : list
        List of files file paths.

    """
    glob_patterns = _check_glob_patterns(glob_patterns)

    filepaths = _get_available_filepaths(raw_dir=raw_dir, station_name=station_name, glob_patterns=glob_patterns)

    # Filter out filepaths if debugging_mode=True
    filepaths = filter_filepaths(filepaths, debugging_mode)

    # Return file list
    return filepaths
