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

import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.utils.directories import (
    check_directory_exist,
    copy_file,
    create_directory,
    create_required_directory,
    ensure_string_path,
    remove_path_trailing_slash,
)
from disdrodb.utils.logger import log_info, log_warning

logger = logging.getLogger(__name__)

####---------------------------------------------------------------------------.
#### Info from file or directory


def _infer_base_dir_from_fpath(path: str) -> str:
    """Return the disdrodb base directory from a file or directory path.

    Assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
        `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    str
        Path of the DISDRODB directory.
    """
    # Retrieve path elements (os-specific)
    p = Path(path)
    list_path_elements = [str(part) for part in p.parts]
    # Retrieve where "DISDRODB" directory occurs
    idx_occurrence = np.where(np.isin(list_path_elements, "DISDRODB"))[0]
    # If DISDRODB directory not present, raise error
    if len(idx_occurrence) == 0:
        raise ValueError(f"The DISDRODB directory is not present in {path}")
    # Find the rightermost occurrence
    right_most_occurrence = max(idx_occurrence)
    # Define the base_dir path
    base_dir = os.path.join(*list_path_elements[: right_most_occurrence + 1])
    return base_dir


def _infer_disdrodb_tree_path(path: str) -> str:
    """Return the directory tree path from the base_dir directory.

    Current assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
        `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    str
        Path inside the DISDRODB archive.
        Format: DISDRODB/<Raw or Processed>/<DATA_SOURCE>/...
    """
    # Retrieve path elements (os-specific)
    p = Path(path)
    list_path_elements = [str(part) for part in p.parts]
    # Retrieve where "DISDRODB" directory occurs
    idx_occurrence = np.where(np.isin(list_path_elements, "DISDRODB"))[0]
    # If DISDRODB directory not present, raise error
    if len(idx_occurrence) == 0:
        raise ValueError(f"The DISDRODB directory is not present in the path '{path}'")
    # Find the rightermost occurrence
    right_most_occurrence = max(idx_occurrence)
    # Define the disdrodb path
    disdrodb_fpath = os.path.join(*list_path_elements[right_most_occurrence:])
    return disdrodb_fpath


def _infer_disdrodb_tree_path_components(path: str) -> list:
    """Return a list with the component of the disdrodb_path.

    Parameters
    ----------
    path : str
        `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    list
        Path element inside the DISDRODB archive.
        Format: ["DISDRODB", <Raw or Processed>, <DATA_SOURCE>, ...]
    """
    # Retrieve disdrodb path
    disdrodb_fpath = _infer_disdrodb_tree_path(path)
    # Retrieve path elements (os-specific)
    p = Path(disdrodb_fpath)
    list_path_elements = [str(part) for part in p.parts]
    return list_path_elements


def _infer_campaign_name_from_path(path: str) -> str:
    """Return the campaign name from a file or directory path.

    Assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
       `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    str
        Name of the campaign.
    """
    list_path_elements = _infer_disdrodb_tree_path_components(path)
    if len(list_path_elements) <= 3:
        raise ValueError(f"Impossible to determine campaign_name from {path}")
    campaign_name = list_path_elements[3]
    return campaign_name


def _infer_data_source_from_path(path: str) -> str:
    """Return the data_source from a file or directory path.

    Assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
       `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    str
        Name of the data source.
    """
    list_path_elements = _infer_disdrodb_tree_path_components(path)
    if len(list_path_elements) <= 2:
        raise ValueError(f"Impossible to determine data_source from {path}")
    data_source = list_path_elements[2]
    return data_source


def _check_campaign_name_is_upper_case(campaign_dir):
    """Check the campaign name of campaign_dir is upper case !"""
    campaign_name = _infer_campaign_name_from_path(campaign_dir)
    upper_campaign_name = campaign_name.upper()
    if campaign_name != upper_campaign_name:
        msg = f"The campaign directory name {campaign_name} must be uppercase: {upper_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)


def _check_data_source_is_upper_case(campaign_dir):
    """Check the data_source name of campaign_dir is upper case !"""
    data_source = _infer_data_source_from_path(campaign_dir)
    upper_data_source = data_source.upper()
    if data_source != upper_data_source:
        msg = f"The data_source directory name {data_source} must be defined uppercase: {upper_data_source}"
        logger.error(msg)
        raise ValueError(msg)


####--------------------------------------------------------------------------.
#### Directory/Filepaths L0A and L0B products


def _get_dataset_min_max_time(ds: xr.Dataset):
    """Retrieves dataset starting and ending time.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    tuple
        (starting_time, ending_time)

    """

    starting_time = ds["time"].values[0]
    ending_time = ds["time"].values[-1]
    return (starting_time, ending_time)


def _get_dataframe_min_max_time(df: pd.DataFrame):
    """Retrieves dataframe starting and ending time.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    tuple
        (starting_time, ending_time)

    """

    starting_time = df["time"].iloc[0]
    ending_time = df["time"].iloc[-1]
    return (starting_time, ending_time)


def get_l0a_dir(processed_dir: str, station_name: str) -> str:
    """Define L0A directory.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0A directory path.
    """
    dir_path = os.path.join(processed_dir, "L0A", station_name)
    return dir_path


def get_l0b_dir(processed_dir: str, station_name: str) -> str:
    """Define L0B directory.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_name : int
        Name of the station

    Returns
    -------
    str
        Path of the L0B directory
    """
    dir_path = os.path.join(processed_dir, "L0B", station_name)
    return dir_path


def get_l0a_fname(df, processed_dir, station_name: str) -> str:
    """Define L0A file name.

    Parameters
    ----------
    df : pd.DataFrame
        L0A DataFrame
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0A file name.
    """
    from disdrodb.l0.standards import PRODUCT_VERSION

    starting_time, ending_time = _get_dataframe_min_max_time(df)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    campaign_name = _infer_campaign_name_from_path(processed_dir).replace(".", "-")
    version = PRODUCT_VERSION
    fname = f"L0A.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.parquet"
    return fname


def get_l0b_fname(ds, processed_dir, station_name: str) -> str:
    """Define L0B file name.

    Parameters
    ----------
    ds : xr.Dataset
        L0B xarray Dataset
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb.l0.standards import PRODUCT_VERSION

    starting_time, ending_time = _get_dataset_min_max_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    campaign_name = _infer_campaign_name_from_path(processed_dir).replace(".", "-")
    version = PRODUCT_VERSION
    fname = f"L0B.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    return fname


def get_l0a_fpath(df: pd.DataFrame, processed_dir: str, station_name: str) -> str:
    """Define L0A file path.

    Parameters
    ----------
    df : pd.DataFrame
        L0A DataFrame.
    processed_dir : str
        Path of the processed directory.
    station_name : str
        Name of the station.

    Returns
    -------
    str
        L0A file path.
    """
    fname = get_l0a_fname(df=df, processed_dir=processed_dir, station_name=station_name)
    dir_path = get_l0a_dir(processed_dir=processed_dir, station_name=station_name)
    fpath = os.path.join(dir_path, fname)
    return fpath


def get_l0b_fpath(ds: xr.Dataset, processed_dir: str, station_name: str, l0b_concat=False) -> str:
    """Define L0B file path.

    Parameters
    ----------
    ds : xr.Dataset
        L0B xarray Dataset.
    processed_dir : str
        Path of the processed directory.
    station_name : str
        ID of the station
    l0b_concat : bool
        If False, the file is specified inside the station directory.
        If True, the file is specified outside the station directory.

    Returns
    -------
    str
        L0B file path.
    """
    dir_path = get_l0b_dir(processed_dir, station_name)
    if l0b_concat:
        dir_path = os.path.dirname(dir_path)
    fname = get_l0b_fname(ds, processed_dir, station_name)
    fpath = os.path.join(dir_path, fname)
    return fpath


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


def _get_file_list_from_glob_pattern(raw_dir: str, station_name, glob_pattern) -> list:
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
    glob_fpath_pattern = os.path.join(data_dir, glob_pattern)
    filepaths = sorted(glob.glob(glob_fpath_pattern))
    return filepaths


def _get_available_filepaths(raw_dir, station_name, glob_patterns):
    # Retrieve filepaths list
    filepaths = [_get_file_list_from_glob_pattern(raw_dir, station_name, pattern) for pattern in glob_patterns]
    filepaths = [x for xs in filepaths for x in xs]  # flatten list

    # Check there are files
    n_files = len(filepaths)
    if n_files == 0:
        glob_fpath_patterns = [os.path.join(raw_dir, pattern) for pattern in glob_patterns]
        raise ValueError(f"No file found at {glob_fpath_patterns}.")
    return filepaths


def _filter_filepaths(filepaths, debugging_mode):
    """Filter out filepaths if debugging_mode=True."""
    if debugging_mode:
        max_files = min(3, len(filepaths))
        filepaths = filepaths[0:max_files]
    return filepaths


def get_raw_file_list(raw_dir, station_name, glob_patterns, verbose=False, debugging_mode=False):
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


def get_l0a_file_list(processed_dir, station_name, debugging_mode):
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
    l0a_dir_path = get_l0a_dir(processed_dir, station_name)
    filepaths = glob.glob(os.path.join(l0a_dir_path, "*.parquet"))

    # Filter out filepaths if debugging_mode=True
    filepaths = _filter_filepaths(filepaths, debugging_mode=debugging_mode)

    # If no file available, raise error
    if len(filepaths) == 0:
        msg = f"No L0A Apache Parquet file is available in {l0a_dir_path}. Run L0A processing first."
        raise ValueError(msg)

    return filepaths


####--------------------------------------------------------------------------.
#### RAW Directory Checks


def _define_metadata_filepath(raw_dir, station_name):
    """Define the filepath of a station metadata YAML file."""
    return os.path.join(raw_dir, "metadata", station_name + ".yml")


def _define_issue_filepath(raw_dir, station_name):
    """Define the filepath of a station issue YAML file."""
    return os.path.join(raw_dir, "issue", station_name + ".yml")


def _define_issue_directory_path(raw_dir):
    """Define the 'issue' directory path."""
    return os.path.join(raw_dir, "issue")


def _define_metadata_directory_path(raw_dir):
    """Define the 'issue' directory path."""
    return os.path.join(raw_dir, "metadata")


def _define_station_directory_path(raw_dir, station_name):
    """Define the data station directory path."""
    return os.path.join(raw_dir, "data", station_name)


def _define_data_directory_path(raw_dir):
    """Define the data directory path."""
    return os.path.join(raw_dir, "data")


def _is_issue_directory_available(raw_dir):
    """Return True if the 'issue' directory is present."""
    return "issue" in os.listdir(raw_dir)


def _is_metadata_directory_available(raw_dir):
    """Return True if the 'metadata' directory is present."""
    return "metadata" in os.listdir(raw_dir)


def _is_data_directory_available(raw_dir):
    """Return True if the 'data' directory is present."""
    return "data" in os.listdir(raw_dir)


def _are_station_directories_available(raw_dir):
    """Return True if within the 'data' directory there are station directories."""
    data_dir = _define_data_directory_path(raw_dir)
    return len(os.listdir(data_dir)) > 0


def _get_available_stations_with_data_directory(raw_dir):
    """Return the name of the station directory in the 'data' directory."""
    data_dir = _define_data_directory_path(raw_dir)
    return os.listdir(data_dir)


def _get_station_raw_filepaths(raw_dir, station_name):
    """Return the filepaths of the files available for a station.

    Note that this function exclude directories !
    """
    station_dir = _define_station_directory_path(raw_dir, station_name)
    paths = glob.glob(os.path.join(station_dir, "*"))
    filepaths = [f for f in paths if os.path.isfile(f)]
    return filepaths


def _get_available_stations_with_metadata_files(raw_dir):
    """Return the name of stations with available metadata YAML files."""
    filepaths = _get_available_metadata_filepaths(raw_dir)
    filenames = [os.path.basename(fpath) for fpath in filepaths]
    station_names = [fname.replace(".yml", "") for fname in filenames]
    return station_names


def _get_available_stations_with_issue_files(raw_dir):
    """Return the name of stations with available issue YAML files."""
    filepaths = _get_available_issue_filepaths(raw_dir)
    filenames = [os.path.basename(fpath) for fpath in filepaths]
    station_names = [fname.replace(".yml", "") for fname in filenames]
    return station_names


def _get_available_metadata_filepaths(raw_dir):
    """Return the filepaths of available metadata YAML files."""
    filepaths = glob.glob(os.path.join(raw_dir, "metadata", "*.yml"))
    return filepaths


def _get_available_issue_filepaths(raw_dir):
    """Return the filepaths of available issue YAML files."""
    filepaths = glob.glob(os.path.join(raw_dir, "issue", "*.yml"))
    return filepaths


def _check_directories_in_raw_dir(raw_dir):
    list_directories = os.listdir(raw_dir)
    if len(list_directories) == 0:
        raise ValueError(f"There are not directories within {raw_dir}")


def _check_presence_data_directory(raw_dir):
    """Check presence of the 'data' directory in the campaign directory."""
    if not _is_data_directory_available(raw_dir):
        raise ValueError(f"'raw_dir' {raw_dir} should have the /data subfolder.")


def _check_presence_stations_directories(raw_dir):
    """Check if there are station directories within 'data'."""
    data_dir = _define_data_directory_path(raw_dir)
    station_names = os.listdir(data_dir)
    if len(station_names) == 0:
        raise ValueError(f"No station directories within {data_dir}")


def _check_presence_of_raw_data(raw_dir):
    """Check presence of raw data in the station directories."""
    # Get name of available stations
    station_names = _get_available_stations_with_data_directory(raw_dir)
    # Count the number of files in each station data directory
    nfiles_per_station = [len(_get_station_raw_filepaths(raw_dir, station_name)) for station_name in station_names]
    # If there is a directory with no data inside, raise error
    idx_no_files = np.where(np.array(nfiles_per_station) == 0)[0]
    if len(idx_no_files) > 0:
        empty_station_dir = [_define_station_directory_path(raw_dir, station_names[idx]) for idx in idx_no_files]
        raise ValueError(f"The following data directories are empty: {empty_station_dir}")


def _check_presence_metadata_directory(raw_dir):
    """Check that the 'metadata' directory exists.

    If the 'metadata' does not exists, it create default metadata files
    for each station present in the 'data' directory.
    """
    from disdrodb.l0.metadata import write_default_metadata

    if not _is_metadata_directory_available(raw_dir):
        # Create metadata directory
        metadata_dir = _define_metadata_directory_path(raw_dir)
        create_directory(metadata_dir)
        # Create default metadata yml file for each station (since the folder didn't existed)
        list_data_station_names = _get_available_stations_with_data_directory(raw_dir)
        list_metadata_fpath = [
            _define_metadata_filepath(raw_dir, station_name) for station_name in list_data_station_names
        ]
        _ = [write_default_metadata(fpath) for fpath in list_metadata_fpath]
        msg = f"'raw_dir' {raw_dir} should have the /metadata subfolder. "
        msg1 = "It has been now created with also empty metadata files to be filled for each station."
        raise ValueError(msg + msg1)


def _check_presence_issue_directory(raw_dir, verbose):
    """If the 'issue' directory does not exist, it creates default issue YAML files."""
    from disdrodb.l0.issue import write_default_issue

    if not _is_issue_directory_available(raw_dir):
        # Create issue directory
        issue_dir = _define_issue_directory_path(raw_dir)
        create_directory(issue_dir)
        # Create issue yml file for each station (since the folder didn't existed)
        list_data_station_names = _get_available_stations_with_data_directory(raw_dir)
        list_issue_fpath = [_define_issue_filepath(raw_dir, station_name) for station_name in list_data_station_names]
        _ = [write_default_issue(fpath) for fpath in list_issue_fpath]
        msg = "The /issue subfolder has been now created to document and then remove timesteps with problematic data."
        log_info(logger, msg, verbose)


def _check_presence_all_metadata_files(raw_dir):
    """Check that the 'metadata' directory contains YAML files.

    The function raise error if there is not a metadata file for each station
    folder present in the 'data' directory.
    """
    from disdrodb.l0.metadata import write_default_metadata

    # Get stations with available metadata
    list_metadata_station_name = _get_available_stations_with_metadata_files(raw_dir)

    # Get stations with available data
    list_data_station_names = _get_available_stations_with_data_directory(raw_dir)

    # Check there is metadata for each station
    # - If missing, create the defaults files and raise an error
    idx_missing_station_data = np.where(np.isin(list_data_station_names, list_metadata_station_name, invert=True))[0]
    if len(idx_missing_station_data) > 0:
        list_missing_station_name = [list_data_station_names[idx] for idx in idx_missing_station_data]
        list_missing_metadata_fpath = [
            _define_metadata_filepath(raw_dir, station_name) for station_name in list_missing_station_name
        ]
        _ = [write_default_metadata(fpath) for fpath in list_missing_metadata_fpath]
        msg = f"The metadata files for the following station_name were missing: {list_missing_station_name}"
        raise ValueError(msg + " Now have been created to be filled.")


def _check_presence_all_issues_files(raw_dir, verbose):
    from disdrodb.l0.issue import write_default_issue

    # Get stations with available issue files
    list_issue_station_name = _get_available_stations_with_issue_files(raw_dir)

    # Get stations with available data
    list_data_station_names = _get_available_stations_with_data_directory(raw_dir)

    # - Check there is issue for each station
    idx_missing_station_data = np.where(np.isin(list_data_station_names, list_issue_station_name, invert=True))[0]
    # - If missing, create the defaults files and raise an error
    if len(idx_missing_station_data) > 0:
        list_missing_station_name = [list_data_station_names[idx] for idx in idx_missing_station_data]
        list_missing_issue_fpath = [
            os.path.join(raw_dir, "issue", station_name + ".yml") for station_name in list_missing_station_name
        ]
        _ = [write_default_issue(fpath) for fpath in list_missing_issue_fpath]
        msg = f"The issue files for the following station_name were missing: {list_missing_station_name}"
        log_warning(logger, msg, verbose)


def _check_no_presence_of_issues_files_without_data(raw_dir, verbose):
    # Get stations with available issue files
    list_issue_station_name = _get_available_stations_with_issue_files(raw_dir)

    # Get stations with available data
    list_data_station_names = _get_available_stations_with_data_directory(raw_dir)

    # - Check not excess issue compared to present stations
    excess_issue_station_namex = np.where(np.isin(list_issue_station_name, list_data_station_names, invert=True))[0]
    if len(excess_issue_station_namex) > 0:
        list_excess_station_name = [list_issue_station_name[idx] for idx in excess_issue_station_namex]
        msg = f"There are the following issue files without corresponding data: {list_excess_station_name}"
        log_warning(logger, msg, verbose)


def _check_no_presence_of_metadata_files_without_data(raw_dir):
    """Check that the 'metadata' directory does not contain excess YAML files."""
    # Get stations with available metadata
    list_metadata_station_name = _get_available_stations_with_metadata_files(raw_dir)

    # Get stations with available data
    list_data_station_names = _get_available_stations_with_data_directory(raw_dir)

    # Check not excess metadata compared to present stations
    idx_excess_metadata_station = np.where(np.isin(list_metadata_station_name, list_data_station_names, invert=True))[0]
    if len(idx_excess_metadata_station) > 0:
        list_excess_station_name = [list_metadata_station_name[idx] for idx in idx_excess_metadata_station]
        print(f"There are the following station metadata files without corresponding data: {list_excess_station_name}")


def _check_valid_metadata(metadata_filepaths):
    """Check all specified metadata files are compliant with DISDRODB standards."""
    from disdrodb.metadata.check_metadata import check_metadata_compliance

    for fpath in metadata_filepaths:
        # Get station info
        base_dir = _infer_base_dir_from_fpath(fpath)
        data_source = _infer_data_source_from_path(fpath)
        campaign_name = _infer_campaign_name_from_path(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")
        # Check compliance
        check_metadata_compliance(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )


def _check_valid_issue_files(filepaths):
    """Check all specified issue files are compliant with DISDRODB standards."""
    from disdrodb.l0.issue import check_issue_file

    _ = [check_issue_file(fpath) for fpath in filepaths]


def _check_raw_dir_is_a_directory(raw_dir):
    """Check that raw_dir is a directory and exists."""
    raw_dir = ensure_string_path(raw_dir, msg="Provide 'raw_dir' as a string", accepth_pathlib=True)
    if not os.path.exists(raw_dir):
        raise ValueError(f"'raw_dir' {raw_dir} directory does not exist.")
    if not os.path.isdir(raw_dir):
        raise ValueError(f"'raw_dir' {raw_dir} is not a directory.")


def _check_raw_dir_data(raw_dir):
    """Check `data` directory in raw campaign directory."""
    # Check the 'data' directory is present
    _check_presence_data_directory(raw_dir)

    # Check presence of station directories
    _check_presence_stations_directories(raw_dir)

    # Check presence of raw data in station directories
    _check_presence_of_raw_data(raw_dir)


def _check_raw_dir_metadata(raw_dir, verbose=True):
    """Check `data` directory in raw campaign directory.

    This function assumes that `the `_check_raw_dir_data`` function
    does not raise errors: a 'data' directory exists, with station subfolders and data files.
    """
    # Check the 'data' directory is present
    _check_presence_data_directory(raw_dir)

    # Check presence of metadata directory
    _check_presence_metadata_directory(raw_dir)

    # Check presence of the expected metadata files
    _check_presence_all_metadata_files(raw_dir)
    _check_no_presence_of_metadata_files_without_data(raw_dir)

    # Check compliance of metadata files
    filepaths = _get_available_metadata_filepaths(raw_dir)
    _check_valid_metadata(filepaths)
    return None


def _check_raw_dir_issue(raw_dir, verbose=True):
    """Check issue yaml files in the raw_dir directory."""

    _check_presence_issue_directory(raw_dir, verbose=verbose)
    _check_presence_all_issues_files(raw_dir, verbose=verbose)
    _check_no_presence_of_issues_files_without_data(raw_dir, verbose=verbose)

    # Check compliance of issue files
    filepaths = _get_available_issue_filepaths(raw_dir)
    _check_valid_issue_files(filepaths)


def check_raw_dir(raw_dir: str, verbose: bool = False) -> None:
    """Check validity of raw_dir.

    Steps:
    1. Check that 'raw_dir' is a valid directory path
    2. Check that 'raw_dir' follows the expect directory structure
    3. Check that each station_name directory contains data
    4. Check that for each station_name the mandatory metadata.yml is specified.
    5. Check that for each station_name the mandatory issue.yml is specified.

    Parameters
    ----------
    raw_dir : str
        Input raw directory
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.

    """
    # Check raw dir is a directory
    _check_raw_dir_is_a_directory(raw_dir)

    # Ensure valid path format
    raw_dir = remove_path_trailing_slash(raw_dir)

    # Check <DATA_SOURCE> and <CAMPAIGN_NAME> are upper case
    _check_data_source_is_upper_case(raw_dir)
    _check_campaign_name_is_upper_case(raw_dir)

    # Check there are directories in raw_dir
    _check_directories_in_raw_dir(raw_dir)

    # Check there is valid /data subfolder
    _check_raw_dir_data(raw_dir)

    # Check there is valid /metadata subfolder
    _check_raw_dir_metadata(raw_dir, verbose=verbose)

    # Check there is valid /issue subfolder
    _check_raw_dir_issue(raw_dir, verbose=verbose)

    return raw_dir


#### -------------------------------------------------------------------------.
#### PROCESSED Directory Checks


def _check_is_inside_processed_directory(processed_dir):
    """Check the path is located within the DISDRODB/Processed directory."""
    # Check is the processed_dir
    if processed_dir.find("DISDRODB/Processed") == -1 and processed_dir.find("DISDRODB\\Processed") == -1:
        msg = "Expecting 'processed_dir' to contain the pattern */DISDRODB/Processed/*. or *\\DISDRODB\\Processed\\*."
        logger.error(msg)
        raise ValueError(msg)


def _check_valid_processed_dir(processed_dir):
    """Check the validity of 'processed_dir'.

    The path must represents this path */DISDRODB/Processed/<DATA_SOURCE>/
    """
    last_component = os.path.basename(processed_dir)
    tree_components = _infer_disdrodb_tree_path_components(processed_dir)
    tree_path = "/".join(tree_components)
    # Check that is not data_source or 'Processed' directory
    if len(tree_components) < 4:
        msg = "Expecting 'processed_dir' to contain the pattern <...>/DISDRODB/Processed/<DATA_SOURCE>/<CAMPAIGN_NAME>."
        msg = msg + f"It only provides {tree_path}"
        logger.error(msg)
        raise ValueError(msg)
    # Check that ends with the campaign_name
    campaign_name = tree_components[3]
    if last_component != campaign_name:
        msg = "Expecting 'processed_dir' to contain the pattern <...>/DISDRODB/Processed/<DATA_SOURCE>/<CAMPAIGN_NAME>."
        msg = msg + f"The 'processed_dir' path {processed_dir} does not end with '{campaign_name}'!"
        logger.error(msg)
        raise ValueError(msg)


def check_processed_dir(processed_dir):
    """Check input, format and validity of the 'processed_dir' directory path.

    Parameters
    ----------
    processed_dir : str
        Path to the campaign directory in the 'DISDRODB/Processed directory tree

    Returns
    -------
    str
        Path of the processed campaign directory
    """
    # Check path type
    processed_dir = ensure_string_path(processed_dir, msg="Provide 'processed_dir' as a string", accepth_pathlib=True)

    # Ensure valid path format
    processed_dir = remove_path_trailing_slash(processed_dir)

    # Check the path is inside the DISDRDB/Processed directory
    _check_is_inside_processed_directory(processed_dir)

    # Check processed_dir is <...>/DISDRODB/Processed/<DATA_SOURCE>/<CAMPAIGN_NAME>
    _check_valid_processed_dir(processed_dir)

    # Check <DATA_SOURCE> and <CAMPAIGN_NAME> are upper case
    _check_data_source_is_upper_case(processed_dir)
    _check_campaign_name_is_upper_case(processed_dir)

    return processed_dir


####---------------------------------------------------------------------------.
#### L0A and L0B directory creation routines


def _check_data_source_consistency(raw_dir: str, processed_dir: str) -> str:
    """Check that 'raw_dir' and 'processed_dir' have same data_source.

    Parameters
    ----------
    raw_dir : str
        Path of the raw campaign directory
    processed_dir : str
        Path of the processed campaign directory

    Returns
    -------
    str
        data_source in capital letter.

    Raises
    ------
    ValueError
        Error if the data_source of the two directory paths does not match.
    """
    raw_data_source = _infer_campaign_name_from_path(raw_dir)
    processed_data_source = _infer_campaign_name_from_path(processed_dir)
    if raw_data_source != processed_data_source:
        msg = f"'raw_dir' and 'processed_dir' must ends with same <CAMPAIGN_NAME>: {raw_data_source}"
        logger.error(msg)
        raise ValueError(msg)
    return raw_data_source.upper()


def _check_campaign_name_consistency(raw_dir: str, processed_dir: str) -> str:
    """Check that 'raw_dir' and 'processed_dir' have same campaign_name.

    Parameters
    ----------
    raw_dir : str
        Path of the raw campaign directory
    processed_dir : str
        Path of the processed campaign directory

    Returns
    -------
    str
        Campaign name in capital letter.

    Raises
    ------
    ValueError
        Error if the campaign_name of the two directory paths does not match.
    """
    raw_campaign_name = _infer_campaign_name_from_path(raw_dir)
    processed_campaign_name = _infer_campaign_name_from_path(processed_dir)
    if raw_campaign_name != processed_campaign_name:
        msg = f"'raw_dir' and 'processed_dir' must ends with same <CAMPAIGN_NAME>: {raw_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)
    return raw_campaign_name.upper()


def _copy_station_metadata(raw_dir: str, processed_dir: str, station_name: str) -> None:
    """Copy the station YAML file from the raw_dir/metadata into processed_dir/metadata

    Parameters
    ----------
    raw_dir : str
        Path of the raw campaign directory
    processed_dir : str
        Path of the processed campaign directory

    Raises
    ------
    ValueError
        Error if the copy fails.
    """
    # Get src and dst metadata directory
    raw_metadata_dir = os.path.join(raw_dir, "metadata")
    processed_metadata_dir = os.path.join(processed_dir, "metadata")
    # Retrieve the metadata fpath in the raw directory
    metadata_fname = f"{station_name}.yml"
    raw_metadata_fpath = os.path.join(raw_metadata_dir, metadata_fname)
    # Check the metadata exists
    if not os.path.isfile(raw_metadata_fpath):
        raise ValueError(f"No metadata available for {station_name} at {raw_metadata_fpath}")
    # Define the destination fpath
    processed_metadata_fpath = os.path.join(processed_metadata_dir, os.path.basename(raw_metadata_fpath))
    # Copy the metadata file
    copy_file(src_fpath=raw_metadata_fpath, dst_fpath=processed_metadata_fpath)
    return None


def _check_pre_existing_station_data(campaign_dir, product, station_name, force=False):
    """Check for pre-existing station data.

    - If force=True, remove all data inside the station folder.
    - If force=False, raise error.

    NOTE: force=False behaviour could be changed to enable updating of missing files.
         This would require also adding code to check whether a downstream file already exist.
    """
    from disdrodb.api.io import _get_list_stations_with_data

    # Get list of available stations
    list_stations = _get_list_stations_with_data(product=product, campaign_dir=campaign_dir)

    # Check if station data are already present
    station_already_present = station_name in list_stations

    # Define the station directory path
    station_dir = os.path.join(campaign_dir, product, station_name)

    # If the station data are already present:
    # - If force=True, remove all data inside the station folder
    # - If force=False, raise error
    if station_already_present:
        # Check is a directory
        check_directory_exist(station_dir)
        # If force=True, remove all the content
        if force:
            # Remove all station directory content
            shutil.rmtree(station_dir)
        else:
            msg = f"The station directory {station_dir} already exists and force=False."
            logger.error(msg)
            raise ValueError(msg)


def create_initial_directory_structure(
    raw_dir,
    processed_dir,
    station_name,
    force,
    product,
    verbose=False,
):
    """Create directory structure for the first L0 DISDRODB product.

    If the input data are raw text files --> product = "L0A"    (run_l0a)
    If the input data are raw netCDF files --> product = "L0B"  (run_l0b_nc)
    """
    from disdrodb.api.io import _get_list_stations_with_data

    # Check inputs
    raw_dir = check_raw_dir(raw_dir=raw_dir, verbose=verbose)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    # Check consistent data_source and campaign name
    _ = _check_data_source_consistency(raw_dir=raw_dir, processed_dir=processed_dir)
    _ = _check_campaign_name_consistency(raw_dir=raw_dir, processed_dir=processed_dir)

    # Get list of available stations (at raw level)
    list_stations = _get_list_stations_with_data(product="RAW", campaign_dir=raw_dir)

    # Check station is available
    if station_name not in list_stations:
        raise ValueError(f"No data available for station {station_name}. Available stations: {list_stations}.")

    # Create required directory (if they don't exists)
    create_required_directory(processed_dir, dir_name="metadata")
    create_required_directory(processed_dir, dir_name="info")
    create_required_directory(processed_dir, dir_name=product)

    # Copy the station metadata
    _copy_station_metadata(raw_dir=raw_dir, processed_dir=processed_dir, station_name=station_name)

    # Remove <product>/<station> directory if force=True
    _check_pre_existing_station_data(
        campaign_dir=processed_dir,
        product=product,
        station_name=station_name,
        force=force,
    )


def create_directory_structure(processed_dir, product, station_name, force, verbose=False):
    """Create directory structure for L0B and higher DISDRODB products."""
    from disdrodb.api.checks import check_product
    from disdrodb.api.io import _get_list_stations_with_data

    # Check inputs
    check_product(product)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    # Check station is available in the target processed_dir directory
    if product == "L0B":
        required_product = "L0A"
        list_stations = _get_list_stations_with_data(product=required_product, campaign_dir=processed_dir)
    else:
        raise NotImplementedError("product {product} not yet implemented.")

    if station_name not in list_stations:
        raise ValueError(
            f"No {required_product} data available for station {station_name}. Available stations: {list_stations}."
        )

    # Create required directory (if they don't exists)
    create_required_directory(processed_dir, dir_name=product)

    # Remove <product>/<station_name> directory if force=True
    _check_pre_existing_station_data(
        campaign_dir=processed_dir,
        product=product,
        station_name=station_name,
        force=force,
    )


####--------------------------------------------------------------------------.
#### DISDRODB L0A Readers


# --> TODO: in L0A processing !
def _read_l0a(fpath: str, verbose: bool = False, debugging_mode: bool = False) -> pd.DataFrame:
    # Log
    msg = f" - Reading L0 Apache Parquet file at {fpath} started."
    log_info(logger, msg, verbose)
    # Open file
    df = pd.read_parquet(fpath)
    if debugging_mode:
        df = df.iloc[0:100]
    # Log
    msg = f" - Reading L0 Apache Parquet file at {fpath} ended."
    log_info(logger, msg, verbose)
    return df


def read_l0a_dataframe(
    fpaths: Union[str, list],
    verbose: bool = False,
    debugging_mode: bool = False,
) -> pd.DataFrame:
    """Read DISDRODB L0A Apache Parquet file(s).

    Parameters
    ----------
    fpaths : str or list
        Either a list or a single filepath .
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        If fpaths is a list, it reads only the first 3 files
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
    if not isinstance(fpaths, (list, str)):
        raise TypeError("Expecting fpaths to be a string or a list of strings.")

    # ----------------------------------------
    # If fpath is a string, convert to list
    if isinstance(fpaths, str):
        fpaths = [fpaths]
    # ---------------------------------------------------
    # - If debugging_mode=True, it reads only the first 3 fpaths
    if debugging_mode:
        fpaths = fpaths[0:3]  # select first 3 fpaths

    # - Define the list of dataframe
    list_df = [_read_l0a(fpath, verbose=verbose, debugging_mode=debugging_mode) for fpath in fpaths]
    # - Concatenate dataframe
    df = concatenate_dataframe(list_df, verbose=verbose)
    # ---------------------------------------------------
    # Return dataframe
    return df
