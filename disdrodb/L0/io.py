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
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd
import importlib.metadata
from typing import Union
from disdrodb.utils.logger import log_info, log_warning
from disdrodb.L0.metadata import read_metadata

logger = logging.getLogger(__name__)

####---------------------------------------------------------------------------.
#### Directory/Filepaths Defaults
def infer_institute_from_fpath(fpath: str) -> str:
    """Infer institue name from file path.

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


# TODO: get_dataframe_min_max_time


def get_L0A_dir(processed_dir: str, station_id: str) -> str:
    """Get L0A directory

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_id : int
        ID of the station

    Returns
    -------
    str
        L0A directory path.
    """
    dir_path = os.path.join(processed_dir, "L0A", station_id)
    return dir_path


def get_L0A_fname(campaign_name: str, station_id: str, suffix: str = "") -> str:
    """build L0A file name.

    Parameters
    ----------
    campaign_name : str
        Name of the campaign.
    station_id : int
        ID of the station
    suffix : int, optional
        suffix, by default ""

    Returns
    -------
    str
        L0A file name.
    """
    if suffix != "":
        suffix = "_" + suffix
    fname = campaign_name + "_s" + station_id + suffix + ".parquet"
    return fname


# TODO: and refactor L0_processing --> remove suffix
#
# def get_L0A_fname(df, processed_dir, station_id: str) -> str:
#     """Define L0A file name.

#     Parameters
#     ----------
#     ds : pd.DataFrame
#         L0A DataFrame
#     processed_dir : str
#         Path of the processed directory
#     station_id : int
#         ID of the station

#     Returns
#     -------
#     str
#         L0B file name.
#     """
#     starting_time, ending_time = get_dataframe_min_max_time(ds)
#     starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
#     ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
#     # production_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#     campaign_name = get_campaign_name(processed_dir).replace(".", "-")
#     institute_name = get_data_source(processed_dir).replace(".", "-")
#     metadata_dict = read_metadata(processed_dir, station_id)
#     sensor_name = metadata_dict.get("sensor_name").replace("_", "-")
#     version = importlib.metadata.version("disdrodb").replace(".", "-")
#     if version == "-VERSION-PLACEHOLDER-":
#         version = "dev"
#     fname = f"DISDRODB.L0A.Raw.{institute_name}.{campaign_name}.{station_id}.{sensor_name}.s{starting_time}.e{ending_time}.{version}.parquet"
#     return fname


def get_L0A_fpath(processed_dir: str, station_id: str, suffix: str = "") -> str:
    """build L0A file path.

    Parameters
    ----------
    campaign_name : str
        Name of the campaign.
    station_id : int
        ID of the station
    suffix : int, optional
        suffix, by default ""

    Returns
    -------
    str
        L0A file path.
    """
    campaign_name = get_campaign_name(processed_dir)
    fname = get_L0A_fname(campaign_name, station_id, suffix=suffix)
    dir_path = get_L0A_dir(processed_dir, station_id)
    fpath = os.path.join(dir_path, fname)
    return fpath


def get_L0B_dir(processed_dir: str, station_id: str) -> str:
    """Build L0B directory

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


def get_L0B_fname(ds, processed_dir, station_id: str) -> str:
    """Define L0B file name.

    Parameters
    ----------
    ds : xr.Dataset
        L0B xarray Dataset
    processed_dir : str
        Path of the processed directory
    station_id : int
        ID of the station

    Returns
    -------
    str
        L0B file name.
    """
    starting_time, ending_time = get_dataset_min_max_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    # production_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # institute_name = get_data_source(processed_dir).replace(".", "-") # TODO: data_source
    campaign_name = get_campaign_name(processed_dir).replace(".", "-")
    metadata_dict = read_metadata(processed_dir, station_id)
    sensor_name = metadata_dict.get("sensor_name").replace("_", "-")

    version = importlib.metadata.version("disdrodb").replace(".", "-")

    if version == "-VERSION-PLACEHOLDER-":
        version = "dev"
    fname = f"DISDRODB.L0B.Raw.{campaign_name}.{station_id}.{sensor_name}.s{starting_time}.e{ending_time}.{version}.nc"
    return fname


def get_L0B_fpath(ds, processed_dir: str, station_id: str) -> str:
    """Define L0B file path.

    Parameters
    ----------
    ds : xr.Dataset
        L0B xarray Dataset
    processed_dir : str
        Path of the processed directory
    station_id : int
        ID of the station

    Returns
    -------
    str
        L0B file path.
    """
    dir_path = get_L0B_dir(processed_dir, station_id)
    fname = get_L0B_fname(ds, processed_dir, station_id)
    fpath = os.path.join(dir_path, fname)
    return fpath


####--------------------------------------------------------------------------.
#### Directory/File Creation/Deletion


def _create_directory(path: str) -> None:
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


####--------------------------------------------------------------------------.
#### Directory checks
def parse_fpath(fpath: str) -> str:
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
        raise TypeError("'parse_fpath' expects a directory/filepath string.")
    if fpath[-1] == "/":
        print("{} should not end with /.".format(fpath))
        fpath = fpath[:-1]

    elif fpath[-1] == "\\":
        print("{} should not end with /.".format(fpath))
        fpath = fpath[:-1]

    return fpath


####--------------------------------------------------------------------------.
#### L0 processing directory checks
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

    Raises
    ------
    TypeError
        Error if not complient.
    """

    from disdrodb.L0.metadata import create_metadata
    from disdrodb.L0.metadata import check_metadata_compliance
    from disdrodb.L0.issue import create_issue_yml
    from disdrodb.L0.issue import check_issue_compliance

    # -------------------------------------------------------------------------.
    # Check input argument
    if not isinstance(raw_dir, str):
        raise TypeError("Provide 'raw_dir' as a string'.")
    if not os.path.exists(raw_dir):
        raise ValueError("'raw_dir' {} directory does not exist.".format(raw_dir))
    if not os.path.isdir(raw_dir):
        raise ValueError("'raw_dir' {} is not a directory.".format(raw_dir))
    # -------------------------------------------------------------------------.
    #### Check there is /data subfolders
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

    # -------------------------------------------------------------------------.
    #### Check there is /metadata subfolders
    metadata_dir = os.path.join(raw_dir, "metadata")
    if "metadata" not in list_subfolders:
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
    # TODO: MISSING IMPLEMENTATION OF check_metadata_compliance
    # -------------------------------------------------------------------------.
    #### Check there is /issue subfolder
    issue_dir = os.path.join(raw_dir, "issue")
    if "issue" not in list_subfolders:
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
    # TODO: MISSING IMPLEMENTATION OF check_issue_compliance
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
            # TODO: maybe remove also the campaign name directory, not only the content of it
            shutil.rmtree(processed_dir)  # TODO: !! TOO DANGEROUS ??? !!!

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

    raw_dir = parse_fpath(raw_dir)
    processed_dir = parse_fpath(processed_dir)
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
        os.makedirs(L0A_folder_path)
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
    fpath: str, lazy: bool = True, verbose: bool = False
) -> Union[pd.DataFrame, dd.DataFrame]:
    # Log
    msg = f" - Reading L0 Apache Parquet file at {fpath} started."
    log_info(logger, msg, verbose)
    # Read
    if lazy:
        df = dd.read_parquet(fpath)
    else:
        df = pd.read_parquet(fpath)
    # Log
    msg = f" - Reading L0 Apache Parquet file at {fpath} ended."
    log_info(logger, msg, verbose)
    return df


def read_L0A_dataframe(
    fpaths: Union[str, list],
    lazy: bool = True,
    verbose: bool = False,
    debugging_mode: bool = False,
) -> Union[pd.DataFrame, dd.DataFrame]:
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
        The default is False.
    lazy : bool
        Whether to read the dataframe lazily with dask.
        If lazy=True, it returns a dask.dataframe.
        If lazy=False, it returns a pandas.DataFrame
        The default is True.

    Returns
    -------
    Union[pd.DataFrame,dd.DataFrame]
        Dataframe

    Raises
    ------
    TypeError
        Error if the reading fails
    """

    from disdrodb.L0.L0A_processing import concatenate_dataframe

    # ----------------------------------------
    # Check fpaths validity
    if not isinstance(fpaths, (list, str)):
        raise TypeError("Expecting fpaths to be a string or a list of strings.")
    # TODO:
    # - CHECK ENDS WITH .parquet
    # ----------------------------------------
    # If list of length 1, convert to string
    if isinstance(fpaths, list) and len(fpaths) == 1:
        fpaths = fpaths[0]
    # ---------------------------------------------------
    # If more than 1 fpath, read and concantenate first
    if isinstance(fpaths, list):
        if debugging_mode:
            fpaths = fpaths[0:3]  # select first 3 fpaths
        list_df = [_read_L0A(fpath, lazy=lazy, verbose=verbose) for fpath in fpaths]
        df = concatenate_dataframe(list_df, verbose=verbose, lazy=lazy)
    # Else read the single file
    else:
        df = _read_L0A(fpaths, lazy=lazy, verbose=verbose)
    # ---------------------------------------------------
    # Return dataframe
    return df


####---------------------------------------------------------------------------.


def check_L0_is_available(
    processed_dir: str, station_id: str, suffix: str = ""
) -> None:
    """Check if the Apache parquet file has been found

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_id : int
        Id of the station.
    suffix : str, optional
        Suffix, by default ""

    Raises
    ------
    ValueError
        Check if the Apache parquet file has not been found
    """
    fpath = get_L0A_fpath(processed_dir, station_id, suffix=suffix)
    if not os.path.exists(fpath):
        msg = f"Need to run L0 processing first. The Apache Parquet file {fpath} is not available."
        logger.exception(msg)
        raise ValueError(msg)
    # Log
    msg = f"Found parquet file: {fpath}"
    logger.info(msg)


def read_L0_data(
    processed_dir: str,
    station_id: str,
    suffix: str = "",
    lazy: bool = True,
    verbose: bool = False,
    debugging_mode: bool = False,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Read L0 Apache Parquet into dataframe.


    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_id : int
        Id of the station.
    suffix : str, optional
        Suffix, by default ""
    lazy : bool, optional
        If True : Dask is used.
        If False : Pandas is used
        by default True
    verbose : bool, optional
        Wheter to verbose the processing.
        The default is False.
    debugging_mode : bool, optional
        If True, return just a subset of total rows, by default False

    Returns
    -------
    Union[pd.DataFrame,dd.DataFrame]
        Dataframe
    """

    # Check L0 is available
    check_L0_is_available(processed_dir, station_id, suffix=suffix)
    # Define fpath
    fpath = get_L0A_fpath(processed_dir, station_id, suffix=suffix)
    # Read L0A Apache Parquet file
    df = _read_L0A(fpath, lazy=lazy, verbose=verbose)
    # Subset dataframe if debugging_mode = True
    if debugging_mode:
        if not lazy:
            df = df.iloc[0:100, :]
        else:
            NotImplementedError
    return df


####--------------------------------------------------------------------------.
