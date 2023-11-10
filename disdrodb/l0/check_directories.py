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
"""Define DISDRODB Checks for Raw and Processed Campaign Directories."""

import glob
import logging
import os

import numpy as np

from disdrodb.api.info import (
    infer_base_dir_from_fpath,
    infer_campaign_name_from_path,
    infer_data_source_from_path,
    infer_disdrodb_tree_path_components,
)
from disdrodb.utils.directories import (
    create_directory,
    ensure_string_path,
    remove_path_trailing_slash,
)
from disdrodb.utils.logger import log_info, log_warning

logger = logging.getLogger(__name__)


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


def _is_metadata_directory_available(campaign_dir):
    """Return True if the 'metadata' directory is present."""
    return "metadata" in os.listdir(campaign_dir)


def _is_metadata_file_available(campaign_dir, station_name):
    metadata_filepath = _define_metadata_filepath(campaign_dir, station_name)
    return os.path.exists(metadata_filepath)


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


#### -------------------------------------------------------------------------.


def _check_campaign_name_is_upper_case(campaign_dir):
    """Check the campaign name of campaign_dir is upper case !"""
    campaign_name = infer_campaign_name_from_path(campaign_dir)
    upper_campaign_name = campaign_name.upper()
    if campaign_name != upper_campaign_name:
        msg = f"The campaign directory name {campaign_name} must be uppercase: {upper_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)


def _check_data_source_is_upper_case(campaign_dir):
    """Check the data_source name of campaign_dir is upper case !"""
    data_source = infer_data_source_from_path(campaign_dir)
    upper_data_source = data_source.upper()
    if data_source != upper_data_source:
        msg = f"The data_source directory name {data_source} must be defined uppercase: {upper_data_source}"
        logger.error(msg)
        raise ValueError(msg)


def check_presence_metadata_directory(campaign_dir):
    """Check that the 'metadata' directory exists.

    If the 'metadata' does not exists, raise an error.
    """
    if not _is_metadata_directory_available(campaign_dir):
        raise ValueError(f"No 'metadata' directory available in {campaign_dir}")


def check_presence_metadata_file(campaign_dir, station_name):
    """Check that the metadata YAML file for the station exists.

    If the metadata YAML file does not exists, raise an error.
    """
    metadata_filepath = _define_metadata_filepath(campaign_dir, station_name)
    if not _is_metadata_file_available(campaign_dir, station_name):
        raise ValueError(f"No metadata YAML file available at {metadata_filepath}")


#### -------------------------------------------------------------------------.
#### Checks for RAW Campaign directory


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
    from disdrodb.metadata.io import write_default_metadata

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
    from disdrodb.metadata.io import write_default_metadata

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
        base_dir = infer_base_dir_from_fpath(fpath)
        data_source = infer_data_source_from_path(fpath)
        campaign_name = infer_campaign_name_from_path(fpath)
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
#### Check for PROCESSED Campaign Directory


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
    tree_components = infer_disdrodb_tree_path_components(processed_dir)
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


#### -------------------------------------------------------------------------.
