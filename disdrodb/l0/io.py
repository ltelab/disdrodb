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
import logging
import os
import shutil
import glob
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union
from disdrodb.utils.logger import log_info, log_warning
from pathlib import Path

logger = logging.getLogger(__name__)

####---------------------------------------------------------------------------.
#### Info from file or directory


def get_disdrodb_dir(path: str) -> str:
    """Return the disdrodb base directory from a file or directory path.

    Current assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

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
    idx_occurence = np.where(np.isin(list_path_elements, "DISDRODB"))[0]
    # If DISDRODB directory not present, raise error
    if len(idx_occurence) == 0:
        raise ValueError(f"The DISDRODB directory is not present in {path}")
    # Find the rightermost occurence
    right_most_occurence = max(idx_occurence)
    # Define the disdrodb_dir path
    disdrodb_dir = os.path.join(*list_path_elements[: right_most_occurence + 1])
    return disdrodb_dir


def get_disdrodb_path(path: str) -> str:
    """Return the path fron the disdrodb_dir directory.

    Current assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
        `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    str
        Path inside the DISDRODB archive.
        Format: DISDRODB/<Raw or Processed>/<data_source>/...
    """
    # Retrieve path elements (os-specific)
    p = Path(path)
    list_path_elements = [str(part) for part in p.parts]
    # Retrieve where "DISDRODB" directory occurs
    idx_occurence = np.where(np.isin(list_path_elements, "DISDRODB"))[0]
    # If DISDRODB directory not present, raise error
    if len(idx_occurence) == 0:
        raise ValueError(f"The DISDRODB directory is not present in {path}")
    # Find the rightermost occurence
    right_most_occurence = max(idx_occurence)
    # Define the disdrodb path
    disdrodb_fpath = os.path.join(*list_path_elements[right_most_occurence:])
    return disdrodb_fpath


def _get_disdrodb_path_components(path: str) -> list:
    """Return a list with the component of the disdrodb_path.

    Parameters
    ----------
    path : str
        `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    list
        Path element inside the DISDRODB archive.
        Format: ["DISDRODB", <Raw or Processed>, <data_source>, ...]
    """
    # Retrieve disdrodb path
    disdrodb_fpath = get_disdrodb_path(path)
    # Retrieve path elements (os-specific)
    p = Path(disdrodb_fpath)
    list_path_elements = [str(part) for part in p.parts]
    return list_path_elements


def get_campaign_name(path: str) -> str:
    """Return the campaign name from a file or directory path.

    Current assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

    Parameters
    ----------
    base_dir : str
       `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    str
        Name of the campaign.
    """
    list_path_elements = _get_disdrodb_path_components(path)
    if len(list_path_elements) <= 3:
        raise ValueError(f"Impossible to determine campaign_name from {path}")
    campaign_name = list_path_elements[3]
    return campaign_name


def get_data_source(path: str) -> str:
    """Return the data_source from a file or directory path.

    Current assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

    Parameters
    ----------
    base_dir : str
       `path` can be a campaign_dir ('raw_dir' or 'processed_dir'), or a DISDRODB file path.

    Returns
    -------
    str
        Name of the campaign.
    """
    list_path_elements = _get_disdrodb_path_components(path)
    if len(list_path_elements) <= 2:
        raise ValueError(f"Impossible to determine data_source from {path}")
    data_source = list_path_elements[2]
    return data_source


####--------------------------------------------------------------------------.
#### Directory/Filepaths L0A and L0B products


def get_dataset_min_max_time(ds: xr.Dataset):
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


def get_dataframe_min_max_time(df: pd.DataFrame):
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


def get_L0A_dir(processed_dir: str, station_name: str) -> str:
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


def get_L0B_dir(processed_dir: str, station_name: str) -> str:
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


def get_L0A_fname(df, processed_dir, station_name: str) -> str:
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

    starting_time, ending_time = get_dataframe_min_max_time(df)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    campaign_name = get_campaign_name(processed_dir).replace(".", "-")
    # metadata_dict = read_metadata(processed_dir, station_name)
    # sensor_name = metadata_dict.get("sensor_name").replace("_", "-")
    version = PRODUCT_VERSION
    fname = f"L0A.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.parquet"
    return fname


def get_L0B_fname(ds, processed_dir, station_name: str) -> str:
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

    starting_time, ending_time = get_dataset_min_max_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    campaign_name = get_campaign_name(processed_dir).replace(".", "-")
    # metadata_dict = read_metadata(processed_dir, station_name)
    # sensor_name = metadata_dict.get("sensor_name").replace("_", "-")
    version = PRODUCT_VERSION
    fname = f"L0B.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    return fname


def get_L0A_fpath(df: pd.DataFrame, processed_dir: str, station_name: str) -> str:
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
    fname = get_L0A_fname(df=df, processed_dir=processed_dir, station_name=station_name)
    dir_path = get_L0A_dir(processed_dir=processed_dir, station_name=station_name)
    fpath = os.path.join(dir_path, fname)
    return fpath


def get_L0B_fpath(ds: xr.Dataset, processed_dir: str, station_name: str, l0b_concat=False) -> str:
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
    dir_path = get_L0B_dir(processed_dir, station_name)
    if l0b_concat:
        dir_path = os.path.dirname(dir_path)
    fname = get_L0B_fname(ds, processed_dir, station_name)
    fpath = os.path.join(dir_path, fname)
    return fpath


####--------------------------------------------------------------------------.
#### List Station Files


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


def check_glob_patterns(patterns: Union[str, list]) -> list:
    """Check if glob patterns are valids."""
    if not isinstance(patterns, (str, list)):
        raise ValueError("'glob_patterns' must be a str or list of strings.")
    if isinstance(patterns, str):
        patterns = [patterns]
    _ = [check_glob_pattern(pattern) for pattern in patterns]
    return patterns


def _get_file_list(raw_dir: str, glob_pattern) -> list:
    """Get the list of files from a directory based on pattern.

    Parameters
    ----------
    raw_dir : str
        Directory of the raw dataset.
    glob_pattern : str
        Pattern to match.

    Returns
    -------
    list
        List of file paths.
    """
    glob_fpath_pattern = os.path.join(raw_dir, glob_pattern)
    list_fpaths = sorted(glob.glob(glob_fpath_pattern))
    return list_fpaths


def get_raw_file_list(raw_dir, station_name, glob_patterns, verbose=False, debugging_mode=False):
    """Get the list of files from a directory based on input parameters.

    Currently concatenates all files provided by the glob patterns.
    In future, this might be modified to enable DISDRODB processing when raw data
    are separated in multiple files.

    Parameters
    ----------
    raw_dir : str
        Directory of the campaign where to search for files.
        Format <..>/DISDRODB/Raw/<data_source>/<campaign_name>
    station_name : str
        ID of the station
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
    # Check glob patterns
    glob_patterns = check_glob_patterns(glob_patterns)

    # Get patterns in the the data directory
    data_dir = os.path.join("data", station_name)
    glob_patterns = [os.path.join(data_dir, pattern) for pattern in glob_patterns]

    # Retrieve filepaths list
    list_fpaths = [_get_file_list(raw_dir, pattern) for pattern in glob_patterns]
    list_fpaths = [x for xs in list_fpaths for x in xs]  # flatten list

    # Check there are files
    n_files = len(list_fpaths)
    if n_files == 0:
        glob_fpath_patterns = [os.path.join(raw_dir, pattern) for pattern in glob_patterns]
        raise ValueError(f"No file found at {glob_fpath_patterns}.")

    # Subset file_list if debugging_mode
    if debugging_mode:
        max_files = min(3, n_files)
        list_fpaths = list_fpaths[0:max_files]

    # Log
    n_files = len(list_fpaths)
    full_dir = os.path.join(raw_dir, data_dir)
    msg = f" - {n_files} files to process in {full_dir}"
    log_info(logger=logger, msg=msg, verbose=verbose)

    # Return file list
    return list_fpaths


def get_l0a_file_list(processed_dir, station_name, debugging_mode):
    """Retrieve L0A files for a give station.

    Parameters
    ----------
    processed_dir : str
        Directory of the campaign where to search for the L0A files.
        Format <..>/DISDRODB/Processed/<data_source>/<campaign_name>
    station_name : str
        ID of the station
    debugging_mode : bool, optional
        If True, it select maximum 3 files for debugging purposes.
        The default is False.

    Returns
    -------
    list_fpaths : list
        List of L0A file paths.

    """
    L0A_dir_path = get_L0A_dir(processed_dir, station_name)
    filepaths = glob.glob(os.path.join(L0A_dir_path, "*.parquet"))

    n_files = len(filepaths)

    # Subset file_list if debugging_mode
    if debugging_mode:
        max_files = min(3, n_files)
        filepaths = filepaths[0:max_files]

    # If no file available, raise error
    if n_files == 0:
        msg = f"No L0A Apache Parquet file is available in {L0A_dir_path}. Run L0A processing first."
        raise ValueError(msg)

    return filepaths


####--------------------------------------------------------------------------.
#### Directory/File Checks/Creation/Deletion


def _check_directory_exist(dir_path):
    """Check if the directory exist."""
    # Check the directory exist
    if not os.path.exists(dir_path):
        raise ValueError(f"{dir_path} directory does not exist.")
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a directory.")


def _create_directory(path: str, exist_ok=True) -> None:
    """Create a directory."""
    if not isinstance(path, str):
        raise TypeError("'path' must be a strig.")
    try:
        os.makedirs(path, exist_ok=exist_ok)
        logger.debug(f"Created directory {path}.")
    except Exception as e:
        dir_name = os.path.basename(path)
        msg = f"Can not create folder {dir_name} inside <path>. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)


def _remove_if_exists(fpath: str, force: bool = False) -> None:
    """Remove file or directory if exists and force=True."""
    # If the file does not exist, do nothing
    if not os.path.exists(fpath):
        return None

    # If the file exist and force=False, raise Error
    if not force:
        msg = f"--force is False and a file already exists at:{fpath}"
        logger.error(msg)
        raise ValueError(msg)

    # If force=True, remove the file.
    try:
        os.remove(fpath)
    except IsADirectoryError:
        try:
            os.rmdir(fpath)
        except OSError:
            try:
                # shutil.rmtree(fpath.rpartition('.')[0])
                for f in glob.glob(fpath + "/*"):
                    try:
                        os.remove(f)
                    except OSError as e:
                        msg = f"Can not delete file {f}, error: {e.strerror}"
                        logger.exception(msg)
                os.rmdir(fpath)
            except:
                msg = f"Something wrong with: {fpath}"
                logger.error(msg)
                raise ValueError(msg)
    logger.info(f"Deleted folder {fpath}")


def _parse_fpath(fpath: str) -> str:
    """Ensure fpath does not end with /.

    Parameters
    ----------
    fpath : str
        Input file path

    Returns
    -------
    str
        Output file path

    Raises
    ------
    TypeError
        Error il file path not compliant
    """

    if not isinstance(fpath, str):
        raise TypeError("'_parse_fpath' expects a directory/filepath string.")
    if fpath[-1] == "/":
        print("{} should not end with /.".format(fpath))
        fpath = fpath[:-1]

    elif fpath[-1] == "\\":
        print("{} should not end with /.".format(fpath))
        fpath = fpath[:-1]

    return fpath


####--------------------------------------------------------------------------.
#### RAW Directory Checks


def _check_raw_dir_input(raw_dir):
    if not isinstance(raw_dir, str):
        raise TypeError("Provide 'raw_dir' as a string'.")
    if not os.path.exists(raw_dir):
        raise ValueError("'raw_dir' {} directory does not exist.".format(raw_dir))
    if not os.path.isdir(raw_dir):
        raise ValueError("'raw_dir' {} is not a directory.".format(raw_dir))


def _check_raw_dir_data_subfolders(raw_dir):
    """Check `data` directory in raw dir."""
    list_subfolders = os.listdir(raw_dir)
    if len(list_subfolders) == 0:
        raise ValueError("There are not subfolders in {}".format(raw_dir))
    if "data" not in list_subfolders:
        raise ValueError("'raw_dir' {} should have the /data subfolder.".format(raw_dir))

    # -------------------------------------------------------------------------.
    #### Check there are subfolders corresponding to station to process
    raw_data_dir = os.path.join(raw_dir, "data")
    list_data_station_name = os.listdir(raw_data_dir)
    if len(list_data_station_name) == 0:
        raise ValueError("No station directories within {}".format(raw_data_dir))

    # -------------------------------------------------------------------------.
    #### Check there are data files in each list_data_station_name
    list_raw_data_station_dir = [os.path.join(raw_data_dir, station_name) for station_name in list_data_station_name]
    list_nfiles_per_station = [len(glob.glob(os.path.join(path, "*"))) for path in list_raw_data_station_dir]
    idx_0_files = np.where(np.array(list_nfiles_per_station) == 0)[0]
    if len(idx_0_files) > 0:
        empty_station_dir = [list_raw_data_station_dir[idx] for idx in idx_0_files]
        raise ValueError("The following data directories are empty: {}".format(empty_station_dir))


def _check_raw_dir_metadata(raw_dir, verbose=True):
    """Check metadata in the raw_dir directory."""
    from disdrodb.l0.metadata import write_default_metadata
    from disdrodb.l0.metadata import check_metadata_compliance

    # Get list of stations
    raw_data_dir = os.path.join(raw_dir, "data")
    list_data_station_name = os.listdir(raw_data_dir)

    # Get metadata directory
    metadata_dir = os.path.join(raw_dir, "metadata")

    # If does not exists
    if "metadata" not in os.listdir(raw_dir):
        # - Create metadata directory
        _create_directory(metadata_dir)
        # - Create default metadata yml file for each station (since the folder didn't existed)
        list_metadata_fpath = [
            os.path.join(metadata_dir, station_name + ".yml") for station_name in list_data_station_name
        ]
        _ = [write_default_metadata(fpath) for fpath in list_metadata_fpath]
        msg = "'raw_dir' {} should have the /metadata subfolder. ".format(raw_dir)
        msg1 = "It has been now created with also empty metadata files to be filled for each station."
        raise ValueError(msg + msg1)

    # -------------------------------------------------------------------------.
    #### Check there are metadata file for each station_name in /metadata
    list_metadata_fpath = glob.glob(os.path.join(metadata_dir, "*.yml"))
    list_metadata_fname = [os.path.basename(fpath) for fpath in list_metadata_fpath]
    list_metadata_station_name = [fname[:-4] for fname in list_metadata_fname]

    # - Check there is metadata for each station
    idx_missing_station_data = np.where(np.isin(list_data_station_name, list_metadata_station_name, invert=True))[0]
    # - If missing, create the defaults files and raise an error
    if len(idx_missing_station_data) > 0:
        list_missing_station_name = [list_data_station_name[idx] for idx in idx_missing_station_data]
        list_missing_metadata_fpath = [
            os.path.join(metadata_dir, station_name + ".yml") for station_name in list_missing_station_name
        ]
        _ = [write_default_metadata(fpath) for fpath in list_missing_metadata_fpath]
        msg = "The metadata files for the following station_name were missing: {}".format(list_missing_station_name)
        raise ValueError(msg + " Now have been created to be filled.")

    # - Check not excess metadata compared to present stations
    idx_excess_metadata_station = np.where(np.isin(list_metadata_station_name, list_data_station_name, invert=True))[0]
    if len(idx_excess_metadata_station) > 0:
        list_excess_station_name = [list_metadata_station_name[idx] for idx in idx_excess_metadata_station]
        print("There are the following metadata files without corresponding data: {}".format(list_excess_station_name))

    # -------------------------------------------------------------------------.
    #### Check metadata compliance
    for fpath in list_metadata_fpath:
        # Get station info
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")
        # Check compliance
        check_metadata_compliance(
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
    return None


def _check_raw_dir_issue(raw_dir, verbose=True):
    """Check issue yaml files in the raw_dir directory."""
    from disdrodb.l0.issue import write_default_issue
    from disdrodb.l0.issue import check_issue_file

    # Get list of stations
    raw_data_dir = os.path.join(raw_dir, "data")
    list_data_station_name = os.listdir(raw_data_dir)
    # Get issue directory
    issue_dir = os.path.join(raw_dir, "issue")
    # If issue directory does not exist
    if "issue" not in os.listdir(raw_dir):
        # - Create issue directory
        _create_directory(issue_dir)
        # - Create issue yml file for each station (since the folder didn't existed)
        list_issue_fpath = [os.path.join(issue_dir, station_name + ".yml") for station_name in list_data_station_name]
        _ = [write_default_issue(fpath) for fpath in list_issue_fpath]
        msg = "The /issue subfolder has been now created to document and then remove timesteps with problematic data."
        logger.info(msg)
    # -------------------------------------------------------------------------.
    #### Check there are issue file for each station_name in /issue
    list_issue_fpath = glob.glob(os.path.join(issue_dir, "*.yml"))
    list_issue_fname = [os.path.basename(fpath) for fpath in list_issue_fpath]
    list_issue_station_name = [fname[:-4] for fname in list_issue_fname]

    # - Check there is issue for each station
    idx_missing_station_data = np.where(np.isin(list_data_station_name, list_issue_station_name, invert=True))[0]
    # - If missing, create the defaults files and raise an error
    if len(idx_missing_station_data) > 0:
        list_missing_station_name = [list_data_station_name[idx] for idx in idx_missing_station_data]
        list_missing_issue_fpath = [
            os.path.join(issue_dir, station_name + ".yml") for station_name in list_missing_station_name
        ]
        _ = [write_default_issue(fpath) for fpath in list_missing_issue_fpath]
        msg = "The issue files for the following station_name were missing: {}".format(list_missing_station_name)
        log_warning(logger, msg, verbose)

    # - Check not excess issue compared to present stations
    excess_issue_station_namex = np.where(np.isin(list_issue_station_name, list_data_station_name, invert=True))[0]
    if len(excess_issue_station_namex) > 0:
        list_excess_station_name = [list_issue_station_name[idx] for idx in excess_issue_station_namex]
        msg = f"There are the following issue files without corresponding data: {list_excess_station_name}"
        log_warning(logger, msg, verbose)

    # -------------------------------------------------------------------------.
    #### Check issue compliance
    _ = [check_issue_file(fpath) for fpath in list_issue_fpath]


def check_raw_dir(raw_dir: str, verbose: bool = False) -> None:
    """Check validity of raw_dir.

    Steps:
    1. Check that 'raw_dir' is a valid directory path
    2. Check that 'raw_dir' follows the expect directory structure
    3. Check that each station_name directory contains data
    4. Check that for each station_name the mandatory metadata.yml is specified.
    4. Check that for each station_name the mandatory issue.yml is specified.

    Parameters
    ----------
    raw_dir : str
        Input raw directory
    verbose : bool, optional
        Wheter to verbose the processing.
        The default is False.

    """
    # -------------------------------------------------------------------------.
    # Check input argument
    _check_raw_dir_input(raw_dir)

    # Ensure valid path format
    raw_dir = _parse_fpath(raw_dir)
    # -------------------------------------------------------------------------.
    # Check there is valid /data subfolder
    _check_raw_dir_data_subfolders(raw_dir)

    # -------------------------------------------------------------------------.
    # Check there is valid /metadata subfolder
    _check_raw_dir_metadata(raw_dir, verbose=verbose)

    # -------------------------------------------------------------------------.
    # Check there is valid /issue subfolder
    _check_raw_dir_issue(raw_dir, verbose=verbose)

    # -------------------------------------------------------------------------.
    return raw_dir


#### -------------------------------------------------------------------------.
#### PROCESSED Directory Checks


def _check_is_processed_dir(processed_dir):
    if not isinstance(processed_dir, str):
        raise TypeError("Provide 'processed_dir' as a string'.")

    # Parse the fpath
    processed_dir = _parse_fpath(processed_dir)

    # Check is the processed_dir
    if processed_dir.find("DISDRODB/Processed") == -1 and processed_dir.find("DISDRODB\\Processed") == -1:
        msg = "Expecting 'processed_dir' to contain the pattern */DISDRODB/Processed/*. or *\DISDRODB\Processed\*."
        logger.error(msg)
        raise ValueError(msg)

    # Check processed_dir does not end with "DISDRODB/Processed"
    # - It must contain also the <campaign_name> directory
    if (
        processed_dir.endswith("Processed")
        or processed_dir.endswith("Processed/")
        or processed_dir.endswith("Processed\\")
    ):
        msg = "Expecting 'processed_dir' to contain the pattern */DISDRODB/Processed/<campaign_name>."
        logger.error(msg)
        raise ValueError(msg)
    return processed_dir


def _check_campaign_name(raw_dir: str, processed_dir: str) -> str:
    """Check that 'raw_dir' and 'processed_dir' have same campaign_name.

    Parameters
    ----------
    raw_dir : str
        Path of the raw directory
    processed_dir : str
        Path of the processed directory

    Returns
    -------
    str
        Campaign name in capital letter

    Raises
    ------
    ValueError
        Error if both paths do not match.
    """
    upper_campaign_name = os.path.basename(raw_dir).upper()
    raw_campaign_name = os.path.basename(raw_dir)
    processed_campaign_name = os.path.basename(processed_dir)
    if raw_campaign_name != processed_campaign_name:
        msg = f"'raw_dir' and 'processed_dir' must ends with same <campaign_name> {upper_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)

    if raw_campaign_name != upper_campaign_name:
        msg = f"'raw_dir' and 'processed_dir' must ends with UPPERCASE <campaign_name> {upper_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)

    return upper_campaign_name


def _create_processed_dir_folder(processed_dir, dir_name):
    """Create directory <dir_name> inside the processed_dir directory."""
    try:
        folder_path = os.path.join(processed_dir, dir_name)
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        msg = f"Can not create folder {dir_name} at {folder_path}. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)


def _copy_station_metadata(raw_dir: str, processed_dir: str, station_name: str) -> None:
    """Copy the station YAML file from the raw_dir/metadata into processed_dir/metadata

    Parameters
    ----------
    raw_dir : str
        Path of the raw directory
    processed_dir : str
        Path of the processed directory

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
    # Try copying the file
    try:
        shutil.copy(raw_metadata_fpath, processed_metadata_fpath)
        msg = f"{metadata_fname} copied at {processed_metadata_fpath}."
        logger.info(msg)
    except Exception as e:
        msg = f"Something went wrong when copying {metadata_fname} into {processed_metadata_dir}.\n The error is: {e}."
        logger.error(msg)
        raise ValueError(msg)
    return None


def _check_pre_existing_station_data(campaign_dir, product_level, station_name, force=False):
    """Check for pre-existing station data.

    - If force=True, remove all data inside the station folder.
    - If force=False, raise error.
    """
    from disdrodb.api.io import _get_list_stations_with_data

    # Get list of available stations
    list_stations = _get_list_stations_with_data(product_level=product_level, campaign_dir=campaign_dir)
    # Check if station data are already present
    station_already_present = station_name in list_stations

    # Define the station directory path
    station_dir = os.path.join(campaign_dir, product_level, station_name)

    # If the station data are already present:
    # - If force=True, remove all data inside the station folder
    # - If force=False, raise error
    # NOTE:
    # - force=False behaviour could be changed to enable updating of missing files.
    #   This would require also adding code to check whether a downstream file already exist.
    if station_already_present:
        # Check is a directory
        _check_directory_exist(station_dir)
        # If force=True, remove all the content
        if force:
            # Remove all station directory content
            shutil.rmtree(station_dir)
        else:
            msg = f"The station directory {station_dir} already exists and force=False."
            logger.error(msg)
            raise ValueError(msg)


def check_processed_dir(processed_dir):
    """Check input, format and validity of the directory path

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory

    Returns
    -------
    str
        Path of the processed directory
    """
    processed_dir = _check_is_processed_dir(processed_dir)
    return processed_dir


# TODO: rename create_initial_directory_structure --> create_initial_directory_structure
def create_initial_directory_structure(raw_dir, processed_dir, station_name, force, verbose=False, product_level="L0A"):
    """Create directory structure for the first L0 DISDRODB product.

    If the input data are raw text files --> product_level = "L0A"    (run_l0a)
    If the input data are raw netCDF files --> product_level = "L0B"  (run_l0b_nc)
    """
    from disdrodb.api.io import _get_list_stations_with_data

    # Check inputs
    raw_dir = check_raw_dir(raw_dir=raw_dir, verbose=verbose)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    # Check valid campaign name
    # - The campaign_name concides between raw and processed dir
    # - The campaign_name is all upper case
    _ = _check_campaign_name(raw_dir=raw_dir, processed_dir=processed_dir)

    # Get list of available stations (at raw level)
    list_stations = _get_list_stations_with_data(product_level="RAW", campaign_dir=raw_dir)
    # Check station is available
    if station_name not in list_stations:
        raise ValueError(f"No data available for station {station_name}. Available stations: {list_stations}.")

    # Create required directory (if they don't exists)
    _create_processed_dir_folder(processed_dir, dir_name="metadata")
    _create_processed_dir_folder(processed_dir, dir_name="info")
    _create_processed_dir_folder(processed_dir, dir_name=product_level)

    # Copy the station metadata
    _copy_station_metadata(raw_dir=raw_dir, processed_dir=processed_dir, station_name=station_name)

    # Remove <product_level>/<station> directory if force=True
    _check_pre_existing_station_data(
        campaign_dir=processed_dir,
        product_level=product_level,
        station_name=station_name,
        force=force,
    )


def create_directory_structure(processed_dir, product_level, station_name, force, verbose=False):
    """Create directory structure for L0B and higher DISDRODB products."""
    from disdrodb.api.io import check_product_level, _get_list_stations_with_data

    # Check inputs
    check_product_level(product_level)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    # Check station is available in the target processed_dir directory
    if product_level == "L0B":
        required_level = "L0A"
        list_stations = _get_list_stations_with_data(product_level=required_level, campaign_dir=processed_dir)
    else:
        raise NotImplementedError("product level {product_level} not yet implemented.")

    if station_name not in list_stations:
        raise ValueError(
            f"No {required_level} data available for station {station_name}. Available stations: {list_stations}."
        )

    # Create required directory (if they don't exists)
    _create_processed_dir_folder(processed_dir, dir_name=product_level)

    # Remove <product_level>/<station_name> directory if force=True
    _check_pre_existing_station_data(
        campaign_dir=processed_dir,
        product_level=product_level,
        station_name=station_name,
        force=force,
    )


####--------------------------------------------------------------------------.
#### DISDRODB L0A Readers
def _read_L0A(fpath: str, verbose: bool = False, debugging_mode: bool = False) -> pd.DataFrame:
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


def read_L0A_dataframe(
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
    list_df = [_read_L0A(fpath, verbose=verbose, debugging_mode=debugging_mode) for fpath in fpaths]
    # - Concatenate dataframe
    df = concatenate_dataframe(list_df, verbose=verbose)
    # ---------------------------------------------------
    # Return dataframe
    return df
