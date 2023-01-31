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
import re
import shutil
import glob
import numpy as np
import pandas as pd
import xarray as xr
import importlib.metadata
from typing import Union
from disdrodb.utils.logger import log_info, log_warning

logger = logging.getLogger(__name__)

####---------------------------------------------------------------------------.
#### Info from filepath
def infer_data_source_from_fpath(fpath: str) -> str:
    """Infer data source from file path.

    Parameters
    ----------
    fpath : str
        Input file path.

    Returns
    -------
    str
        Name of the institute.
    """
    path_pattern = r"(\\|\/)"
    idx_start = fpath.rfind("DISDRODB")
    disdrodb_fpath = fpath[idx_start:]
    institute = re.split(path_pattern, disdrodb_fpath)[4]
    return institute


def infer_campaign_from_fpath(fpath: str) -> str:
    """Infer campaign name from file path.

    Parameters
    ----------
    fpath : str
        Input file path.

    Returns
    -------
    str
        Name of the campaign.
    """
    path_pattern = r"(\\|\/)"
    idx_start = fpath.rfind("DISDRODB")
    disdrodb_fpath = fpath[idx_start:]
    campaign = re.split(path_pattern, disdrodb_fpath)[6]
    return campaign


def infer_station_id_from_fpath(fpath: str) -> str:
    """
    Get the station ID from the path of the input raw data.

    Parameters
    ----------
    fpath : str
        Path of the raw file.

    Returns
    -------
    str
        Station ID
    """
    path_pattern = r"(\\|\/)"
    idx_start = fpath.rfind("DISDRODB")
    disdrodb_fpath = fpath[idx_start:]
    list_path_elements = re.split(path_pattern, disdrodb_fpath)
    station_id = list_path_elements[8]
    # Optional strip .yml if fpath point to YAML file
    station_id.strip(".yml")
    return station_id


def get_disdrodb_dir(base_dir: str) -> str:
    """Return the disdrodb base directory from 'raw_dir' or 'processed_dir' paths.

    Parameters
    ----------
    base_dir : str
        Path 'raw_dir' or 'processed_dir' directory.

    Returns
    -------
    str
        Path of the DISDRODB directory.
    """
    idx_start = base_dir.rfind("DISDRODB")
    disdrodb_dir = os.path.join(base_dir[:idx_start], "DISDRODB")
    return disdrodb_dir


def get_campaign_name(base_dir: str) -> str:
    """Return the campaign name from 'raw_dir' or 'processed_dir' paths.

    Parameters
    ----------
    base_dir : str
        Path 'raw_dir' or 'processed_dir' directory.

    Returns
    -------
    str
        Name of the campaign.
    """
    path_pattern = r"(\\|\/)"
    idx_start = base_dir.rfind("DISDRODB")
    disdrodb_fpath = base_dir[idx_start:]
    list_path_elements = re.split(path_pattern, disdrodb_fpath)
    campaign_name = list_path_elements[-1].upper()
    return campaign_name


def get_data_source(base_dir: str) -> str:
    """Retrieves the data source from 'raw_dir' or processed_dir' paths

    Parameters
    ----------
    base_dir : str
        Input paths

    Returns
    -------
    str
        Name of the data source
    """

    path_pattern = r"(\\|\/)"
    idx_start = base_dir.rfind("DISDRODB")
    disdrodb_fpath = base_dir[idx_start:]
    list_path_elements = re.split(path_pattern, disdrodb_fpath)
    institute_name = list_path_elements[-3].upper()
    return institute_name


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


def get_L0A_dir(processed_dir: str, station_id: str) -> str:
    """Define L0A directory.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_id : str
        ID of the station

    Returns
    -------
    str
        L0A directory path.
    """
    dir_path = os.path.join(processed_dir, "L0A", station_id)
    return dir_path


def get_L0B_dir(processed_dir: str, station_id: str) -> str:
    """Define L0B directory.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_id : int
        ID of the station

    Returns
    -------
    str
        Path of the L0B directory
    """
    dir_path = os.path.join(processed_dir, "L0B", station_id)
    return dir_path


def get_L0A_fname(df, processed_dir, station_id: str) -> str:
    """Define L0A file name.

    Parameters
    ----------
    df : pd.DataFrame
        L0A DataFrame
    processed_dir : str
        Path of the processed directory
    station_id : str
        ID of the station

    Returns
    -------
    str
        L0A file name.
    """
    starting_time, ending_time = get_dataframe_min_max_time(df)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    campaign_name = get_campaign_name(processed_dir).replace(".", "-")
    # metadata_dict = read_metadata(processed_dir, station_id)
    # sensor_name = metadata_dict.get("sensor_name").replace("_", "-")
    version = importlib.metadata.version("disdrodb").replace(".", "-")
    if version == "-VERSION-PLACEHOLDER-":
        version = "dev"
    fname = f"DISDRODB.L0A.{campaign_name}.{station_id}.s{starting_time}.e{ending_time}.{version}.parquet"
    return fname


def get_L0B_fname(ds, processed_dir, station_id: str) -> str:
    """Define L0B file name.

    Parameters
    ----------
    ds : xr.Dataset
        L0B xarray Dataset
    processed_dir : str
        Path of the processed directory
    station_id : str
        ID of the station

    Returns
    -------
    str
        L0B file name.
    """
    starting_time, ending_time = get_dataset_min_max_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    campaign_name = get_campaign_name(processed_dir).replace(".", "-")
    # metadata_dict = read_metadata(processed_dir, station_id)
    # sensor_name = metadata_dict.get("sensor_name").replace("_", "-")
    version = importlib.metadata.version("disdrodb").replace(".", "-")
    if version == "-VERSION-PLACEHOLDER-":
        version = "dev"
    fname = f"DISDRODB.L0B.{campaign_name}.{station_id}.s{starting_time}.e{ending_time}.{version}.nc"
    return fname


def get_L0A_fpath(df: pd.DataFrame, processed_dir: str, station_id: str) -> str:
    """Define L0A file path.

    Parameters
    ----------
    df : pd.DataFrame
        L0A DataFrame.
    processed_dir : str
        Path of the processed directory.
    station_id : str
        ID of the station.

    Returns
    -------
    str
        L0A file path.
    """
    fname = get_L0A_fname(df=df, processed_dir=processed_dir, station_id=station_id)
    dir_path = get_L0A_dir(processed_dir=processed_dir, station_id=station_id)
    fpath = os.path.join(dir_path, fname)
    return fpath


def get_L0B_fpath(
    ds: xr.Dataset, processed_dir: str, station_id: str, single_netcdf=False
) -> str:
    """Define L0B file path.

    Parameters
    ----------
    ds : xr.Dataset
        L0B xarray Dataset.
    processed_dir : str
        Path of the processed directory.
    station_id : str
        ID of the station
    single_netcdf : bool
        If False, the file is specified inside the station directory.
        If True, the file is specified outside the station directory.

    Returns
    -------
    str
        L0B file path.
    """
    dir_path = get_L0B_dir(processed_dir, station_id)
    if single_netcdf:
        dir_path = os.path.dirname(dir_path)
    fname = get_L0B_fname(ds, processed_dir, station_id)
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


def get_raw_file_list(
    raw_dir, station_id, glob_patterns, verbose=False, debugging_mode=False
):
    """Get the list of files from a directory based on input parameters.

    Currently concatenates all files provided by the glob patterns.
    In future, this might be modified to enable DISDRODB processing when raw data
    are separated in multiple files.

    Parameters
    ----------
    raw_dir : str
        Directory of the campaign where to search for files.
        Format <..>/DISDRODB/Raw/<data_source>/<campaign_name>
    station_id : str
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
    data_dir = os.path.join("data", station_id)
    glob_patterns = [os.path.join(data_dir, pattern) for pattern in glob_patterns]

    # Retrieve filepaths list
    list_fpaths = [_get_file_list(raw_dir, pattern) for pattern in glob_patterns]
    list_fpaths = [x for xs in list_fpaths for x in xs]  # flatten list

    # Check there are files
    n_files = len(list_fpaths)
    if n_files == 0:
        glob_fpath_patterns = [
            os.path.join(raw_dir, pattern) for pattern in glob_patterns
        ]
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


def get_l0a_file_list(processed_dir, station_id, debugging_mode):
    """Retrieve L0A files for a give station.

    Parameters
    ----------
    processed_dir : str
        Directory of the campaign where to search for the L0A files.
        Format <..>/DISDRODB/Processed/<data_source>/<campaign_name>
    station_id : str
        ID of the station
    debugging_mode : bool, optional
        If True, it select maximum 3 files for debugging purposes.
        The default is False.

    Returns
    -------
    list_fpaths : list
        List of L0A file paths.

    """
    L0A_dir_path = get_L0A_dir(processed_dir, station_id)
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


def _create_directory(path: str) -> None:
    """Create a directory."""
    if not isinstance(path, str):
        raise TypeError("'path' must be a strig.")
    try:
        os.makedirs(path)
        logger.debug(f"Created directory {path}.")
    except FileExistsError:
        logger.debug(f"Directory {path} already exists.")
        pass
    except (Exception) as e:
        dir_name = os.path.basename(path)
        msg = f"Can not create folder {dir_name} inside <path>. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)


def _remove_if_exists(fpath: str, force: bool = False) -> None:
    """Remove file or directory if exists and force=True."""
    if os.path.exists(fpath):
        if not force:
            msg = f"--force is False and a file already exists at:{fpath}"
            logger.error(msg)
            raise ValueError(msg)
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
#### L0 processing directory checks


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
        raise ValueError(
            "'raw_dir' {} should have the /data subfolder.".format(raw_dir)
        )

    # -------------------------------------------------------------------------.
    #### Check there are subfolders corresponding to station to process
    raw_data_dir = os.path.join(raw_dir, "data")
    list_data_station_id = os.listdir(raw_data_dir)
    if len(list_data_station_id) == 0:
        raise ValueError("No station directories within {}".format(raw_data_dir))

    # -------------------------------------------------------------------------.
    #### Check there are data files in each list_data_station_id
    list_raw_data_station_dir = [
        os.path.join(raw_data_dir, station_id) for station_id in list_data_station_id
    ]
    list_nfiles_per_station = [
        len(glob.glob(os.path.join(path, "*"))) for path in list_raw_data_station_dir
    ]
    idx_0_files = np.where(np.array(list_nfiles_per_station) == 0)[0]
    if len(idx_0_files) > 0:
        empty_station_dir = [list_raw_data_station_dir[idx] for idx in idx_0_files]
        raise ValueError(
            "The following data directories are empty: {}".format(empty_station_dir)
        )


def _check_raw_dir_metadata(raw_dir, verbose=True):
    """Check metadata in the raw_dir directory."""
    from disdrodb.L0.metadata import create_metadata
    from disdrodb.L0.metadata import check_metadata_compliance

    # Get list of stations
    raw_data_dir = os.path.join(raw_dir, "data")
    list_data_station_id = os.listdir(raw_data_dir)

    # Get metadata directory
    metadata_dir = os.path.join(raw_dir, "metadata")

    # If does not exists
    if "metadata" not in os.listdir(raw_dir):
        # - Create metadata directory
        _create_directory(metadata_dir)
        # - Create default metadata yml file for each station (since the folder didn't existed)
        list_metadata_fpath = [
            os.path.join(metadata_dir, station_id + ".yml")
            for station_id in list_data_station_id
        ]
        _ = [create_metadata(fpath) for fpath in list_metadata_fpath]
        msg = "'raw_dir' {} should have the /metadata subfolder. ".format(raw_dir)
        msg1 = "It has been now created with also empty metadata files to be filled for each station."
        raise ValueError(msg + msg1)

    # -------------------------------------------------------------------------.
    #### Check there are metadata file for each station_id in /metadata
    list_metadata_fpath = glob.glob(os.path.join(metadata_dir, "*.yml"))
    list_metadata_fname = [os.path.basename(fpath) for fpath in list_metadata_fpath]
    list_metadata_station_id = [fname[:-4] for fname in list_metadata_fname]

    # - Check there is metadata for each station
    missing_data_station_idx = np.where(
        np.isin(list_data_station_id, list_metadata_station_id, invert=True)
    )[0]
    # - If missing, create the defaults files and raise an error
    if len(missing_data_station_idx) > 0:
        list_missing_station_id = [
            list_data_station_id[idx] for idx in missing_data_station_idx
        ]
        list_missing_metadata_fpath = [
            os.path.join(metadata_dir, station_id + ".yml")
            for station_id in list_missing_station_id
        ]
        _ = [create_metadata(fpath) for fpath in list_missing_metadata_fpath]
        msg = "The metadata files for the following station_id were missing: {}".format(
            list_missing_station_id
        )
        raise ValueError(msg + " Now have been created to be filled.")

    # - Check not excess metadata compared to present stations
    excess_metadata_station_idx = np.where(
        np.isin(list_metadata_station_id, list_data_station_id, invert=True)
    )[0]
    if len(excess_metadata_station_idx) > 0:
        list_excess_station_id = [
            list_metadata_station_id[idx] for idx in excess_metadata_station_idx
        ]
        print(
            "There are the following metadata files without corresponding data: {}".format(
                list_excess_station_id
            )
        )

    # -------------------------------------------------------------------------.
    #### Check metadata compliance
    _ = [check_metadata_compliance(fpath) for fpath in list_metadata_fpath]
    return None


def _check_raw_dir_issue(raw_dir, verbose=True):
    """Check issue yaml files in the raw_dir directory."""
    from disdrodb.L0.issue import create_issue_yml
    from disdrodb.L0.issue import check_issue_compliance

    # Get list of stations
    raw_data_dir = os.path.join(raw_dir, "data")
    list_data_station_id = os.listdir(raw_data_dir)
    # Get issue directory
    issue_dir = os.path.join(raw_dir, "issue")
    # If issue directory does not exist
    if "issue" not in os.listdir(raw_dir):
        # - Create issue directory
        _create_directory(issue_dir)
        # - Create issue yml file for each station (since the folder didn't existed)
        list_issue_fpath = [
            os.path.join(issue_dir, station_id + ".yml")
            for station_id in list_data_station_id
        ]
        _ = [create_issue_yml(fpath) for fpath in list_issue_fpath]
        msg = "The /issue subfolder has been now created to document and then remove timesteps with problematic data."
        logger.info(msg)
    # -------------------------------------------------------------------------.
    #### Check there are issue file for each station_id in /issue
    list_issue_fpath = glob.glob(os.path.join(issue_dir, "*.yml"))
    list_issue_fname = [os.path.basename(fpath) for fpath in list_issue_fpath]
    list_issue_station_id = [fname[:-4] for fname in list_issue_fname]

    # - Check there is issue for each station
    missing_data_station_idx = np.where(
        np.isin(list_data_station_id, list_issue_station_id, invert=True)
    )[0]
    # - If missing, create the defaults files and raise an error
    if len(missing_data_station_idx) > 0:
        list_missing_station_id = [
            list_data_station_id[idx] for idx in missing_data_station_idx
        ]
        list_missing_issue_fpath = [
            os.path.join(issue_dir, station_id + ".yml")
            for station_id in list_missing_station_id
        ]
        _ = [create_issue_yml(fpath) for fpath in list_missing_issue_fpath]
        msg = "The issue files for the following station_id were missing: {}".format(
            list_missing_station_id
        )
        log_warning(logger, msg, verbose)

    # - Check not excess issue compared to present stations
    excess_issue_station_idx = np.where(
        np.isin(list_issue_station_id, list_data_station_id, invert=True)
    )[0]
    if len(excess_issue_station_idx) > 0:
        list_excess_station_id = [
            list_issue_station_id[idx] for idx in excess_issue_station_idx
        ]
        msg = f"There are the following issue files without corresponding data: {list_excess_station_id}"
        log_warning(logger, msg, verbose)

    # -------------------------------------------------------------------------.
    #### Check issue compliance
    _ = [check_issue_compliance(fpath) for fpath in list_issue_fpath]


def check_raw_dir(raw_dir: str, verbose: bool = False) -> None:
    """Check validity of raw_dir.

    Steps:
    1. Check that 'raw_dir' is a valid directory path
    2. Check that 'raw_dir' follows the expect directory structure
    3. Check that each station_id directory contains data
    4. Check that for each station_id the mandatory metadata are specified.

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


def check_processed_dir(processed_dir: str, force: bool = False) -> None:
    """Check that 'processed_dir' is a valid directory path.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    force : bool, optional
        If True, overwrite existing data into processed directory.
        If False, raise an error if there are already data into processed directory.

    Raises
    ------
    TypeError
        Error if path pattern not respected or can not be created.


    """

    if not isinstance(processed_dir, str):
        raise TypeError("Provide 'processed_dir' as a string'.")
    # ------------------------------
    # Check processed_dir has "DISDRODB/Processed" to avoid deleting precious stuffs
    if (
        processed_dir.find("DISDRODB/Processed") == -1
        and processed_dir.find("DISDRODB\\Processed") == -1
    ):
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

    # ------------------------------
    # If forcing overwriting
    if force:
        # Check processed_dir is a directory before removing the content
        if os.path.exists(processed_dir):
            if not os.path.isdir(processed_dir):
                msg = f"'processed_dir' {processed_dir} already exist but is not a directory."
                logger.error(msg)
                raise ValueError(msg)
            # Remove content of existing processed_dir
            # TODO: https://github.com/ltelab/disdrodb/issues/113
            # - if l0a_processing=False, remove only L0B directory ! --> Otherwise then no data to process
            # - if l0a_processing=True, remove as it done now
            # --> Require adding such argumments to this function and create_directory_structure
            shutil.rmtree(processed_dir)

    # ------------------------------
    # If avoiding overwriting
    if not force:
        if os.path.exists(processed_dir):
            msg = f"'processed_dir' {processed_dir} already exists and force=False."
            logger.error(msg)
            raise ValueError(msg)

    # ------------------------------
    # Recreate processed_dir
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    else:
        msg = "Please report the BUG. This should not happen."
        logger.error(msg)
        raise ValueError(msg)


def check_campaign_name(raw_dir: str, processed_dir: str) -> str:
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


def check_directories(raw_dir: str, processed_dir: str, force: bool = False) -> tuple:
    """Check that the specified directories respect the standards.

    Parameters
    ----------
    raw_dir : str
        Path of the raw directory
    processed_dir : str
        Path of the processed directory
    force : bool, optional
        If True, overwrite existing data into processed directory.
        If False, raise an error if there are already data into processed directory.

    Returns
    -------
    tuple
        raw directory and processed directory
    """

    raw_dir = _parse_fpath(raw_dir)
    processed_dir = _parse_fpath(processed_dir)
    check_raw_dir(raw_dir)
    check_processed_dir(processed_dir, force=force)
    check_campaign_name(raw_dir, processed_dir)
    return raw_dir, processed_dir


def check_metadata_dir(processed_path: str) -> None:
    """Create metadata folder into process directory.

    Parameters
    ----------
    processed_path : str
        Path of the processed directory

    Raises
    ------
    FileNotFoundError
        Error metadat already existed or can not be created.
    """
    # Create metadata folder
    try:
        metadata_folder_path = os.path.join(processed_path, "metadata")
        os.makedirs(metadata_folder_path)
        logger.debug(f"Created {metadata_folder_path}.")
    except FileExistsError:
        logger.debug(f"{metadata_folder_path} already existed.")
        pass
    except (Exception) as e:
        msg = f"Folder metadata can not be created in {metadata_folder_path}>. \n The error is: {e}."
        logger.exception(msg)
        raise FileNotFoundError(msg)


def copy_metadata_from_raw_dir(raw_dir: str, processed_dir: str) -> None:
    """Copy yaml files in raw_dir/metadata into processed_dir/metadata

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
    # Retrieve metadata fpaths in raw directory
    raw_metadata_fpaths = glob.glob(os.path.join(raw_metadata_dir, "*.yml"))
    # Copy all metadata yml files into the "processed" folder
    for raw_metadata_fpath in raw_metadata_fpaths:
        # Check if is a files
        if os.path.isfile(raw_metadata_fpath):
            metadata_fname = os.path.basename(raw_metadata_fpath)
            processed_metadata_fpath = os.path.join(
                processed_metadata_dir, metadata_fname
            )
            try:
                # Copy every file
                shutil.copy(raw_metadata_fpath, processed_metadata_fpath)
                msg = f"{metadata_fname} copied into {processed_metadata_dir}."
                logger.info(msg)
            except (Exception) as e:
                msg = f"Something went wrong when copying {metadata_fname} into {processed_metadata_dir}.\n The error is: {e}."
                logger.error(msg)
        else:
            msg = f"Cannot copy {metadata_fname} into {processed_metadata_dir}."
            logger.error(msg)
            raise ValueError(msg)
    metadata_fnames = [
        os.path.basename(metadata_fpath) for metadata_fpath in raw_metadata_fpaths
    ]
    msg = f"The metadata of stations ({metadata_fnames}) have been copied into {processed_metadata_dir}."
    logger.info(msg)


####--------------------------------------------------------------------------.
#### L0 processing directory structure


def create_directory_structure(raw_dir: str, processed_dir: str) -> None:
    """Create directory structure for L0A and L0B processing.

    Parameters
    ----------
    raw_dir : str
        Path of the raw directory
    processed_dir : str
        Path of the processed directory

    Raises
    ------
    FileNotFoundError
        Error is folder structure can not be created.
    """

    # -----------------------------------------------------.
    #### Create metadata folder inside processed_dir
    check_metadata_dir(processed_dir)
    copy_metadata_from_raw_dir(raw_dir, processed_dir)

    # -----------------------------------------------------.
    #### Create info folder inside processed_dir
    try:
        info_folder_path = os.path.join(processed_dir, "info")
        os.makedirs(info_folder_path)
        logger.debug(f"Created {info_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {info_folder_path}")
        pass
    except (Exception) as e:
        msg = f"Can not create folder metadata inside <info_folder_path>. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)

    # -----------------------------------------------------.
    #### Create L0A folder inside processed_dir
    try:
        L0A_folder_path = os.path.join(processed_dir, "L0A")
        os.makedirs(L0A_folder_path, exist_ok=True)
        logger.debug(f"Created {L0A_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {L0A_folder_path}")
        pass
    except (Exception) as e:
        msg = f"Can not create folder L0A inside {processed_dir}. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)

    # -----------------------------------------------------.
    #### Create L0B folder inside processed_dir
    try:
        L0B_folder_path = os.path.join(processed_dir, "L0B")
        os.makedirs(L0B_folder_path)
        logger.debug(f"Created {L0B_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {L0B_folder_path}")
        pass
    except (Exception) as e:
        msg = f"Can not create folder L0B inside {processed_dir}. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)
    return


####--------------------------------------------------------------------------.
#### DISDRODB L0A Readers
def _read_L0A(
    fpath: str, verbose: bool = False, debugging_mode: bool = False
) -> pd.DataFrame:
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

    from disdrodb.L0.L0A_processing import concatenate_dataframe

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
    list_df = [
        _read_L0A(fpath, verbose=verbose, debugging_mode=debugging_mode)
        for fpath in fpaths
    ]
    # - Concatenate dataframe
    df = concatenate_dataframe(list_df, verbose=verbose)
    # ---------------------------------------------------
    # Return dataframe
    return df
