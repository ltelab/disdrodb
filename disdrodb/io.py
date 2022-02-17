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
import dask.dataframe as dd

from disdrodb.metadata import create_metadata
from disdrodb.metadata import check_metadata_compliance

logger = logging.getLogger(__name__)

####---------------------------------------------------------------------------.
#### Directory/Filepaths Defaults


def get_campaign_name(base_dir):
    """Return the campaign name from 'raw_dir' or processed_dir' paths."""
    base_dir = parse_fpath(base_dir)
    campaign_name = os.path.basename(base_dir).upper()
    return campaign_name


def get_L0_fname(campaign_name, station_id, suffix=""):
    if suffix != "":
        suffix = "_" + suffix
    fname = campaign_name + "_s" + station_id + suffix + ".parquet"
    return fname


def get_L0_fpath(processed_dir, station_id, suffix=""):
    campaign_name = get_campaign_name(processed_dir)
    fname = get_L0_fname(campaign_name, station_id, suffix=suffix)
    fpath = os.path.join(processed_dir, "L0", fname)
    return fpath


def get_L1_netcdf_fname(campaign_name, station_id, suffix=""):
    if suffix != "":
        suffix = "_" + suffix
    fname = campaign_name + "_s" + station_id + suffix + ".nc"
    return fname


def get_L1_netcdf_fpath(processed_dir, station_id, suffix=""):
    campaign_name = get_campaign_name(processed_dir)
    fname = get_L1_netcdf_fname(campaign_name, station_id, suffix=suffix)
    fpath = os.path.join(processed_dir, "L1", fname)
    return fpath


def get_L1_zarr_fname(campaign_name, station_id, suffix=""):
    if suffix != "":
        suffix = "_" + suffix
    fname = campaign_name + "_s" + station_id + suffix + ".zarr"
    return fname


def get_L1_zarr_fpath(processed_dir, station_id, suffix=""):
    campaign_name = get_campaign_name(processed_dir)
    fname = get_L1_zarr_fname(campaign_name, station_id, suffix=suffix)
    fpath = os.path.join(processed_dir, "L1", fname)
    return fpath


####--------------------------------------------------------------------------.
#### Directory/File Creation/Deletion


def _create_directory(path):
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


def _remove_if_exists(fpath, force=False):
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
                except (Exception) as e:
                    msg = f"Something wrong with: {fpath}"
                    logger.error(msg)
                    raise ValueError(msg)
        logger.info(f"Deleted folder {fpath}")


####--------------------------------------------------------------------------.
#### Directory checks


def parse_fpath(fpath):
    """Ensure fpath does not end with /."""
    if not isinstance(fpath, str):
        raise TypeError("'parse_fpath' expects a directory/filepath string.")
    if fpath[-1] == "/":
        print("{} should not end with /.".format(fpath))
        fpath = fpath[:-1]
    return fpath


def check_raw_dir(raw_dir):
    """Check validity of raw_dir.

    Steps:
    1. Check that 'raw_dir' is a valid directory path
    2. Check that 'raw_dir' follows the expect directory structure
    3. Check that each station_id directory contains data
    4. Check that for each station_id the mandatory metadata are specified.

    """
    if not isinstance(raw_dir, str):
        raise TypeError("Provide 'raw_dir' as a string'.")
    if not os.path.exists(raw_dir):
        raise ValueError("'raw_dir' {} directory does not exist.".format(raw_dir))
    if not os.path.isdir(raw_dir):
        raise ValueError("'raw_dir' {} is not a directory.".format(raw_dir))
    # -------------------------------------------------------------------------.
    # Check there is /data subfolders
    list_subfolders = os.listdir(raw_dir)
    if len(list_subfolders) == 0:
        raise ValueError("There are not subfolders in {}".format(raw_dir))
    if "data" not in list_subfolders:
        raise ValueError(
            "'raw_dir' {} should have the /data subfolder.".format(raw_dir)
        )

    # Check there are subfolders corresponding to station to process
    raw_data_dir = os.path.join(raw_dir, "data")
    list_data_station_id = os.listdir(raw_data_dir)
    if len(list_data_station_id) == 0:
        raise ValueError("No station directories within {}".format(raw_data_dir))

    # Check there are data files in each list_data_station_id
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
    # Check there is /metadata subfolders
    metadata_dir = os.path.join(raw_dir, "metadata")
    if "metadata" not in list_subfolders:
        # - Create directory with empty metadata files to be filled.
        _create_directory(metadata_dir)
        list_metadata_fpath = [
            os.path.join(metadata_dir, station_id + ".yml")
            for station_id in list_data_station_id
        ]
        _ = [create_metadata(fpath) for fpath in list_metadata_fpath]
        msg = "'raw_dir' {} should have the /metadata subfolder. ".format(raw_dir)
        msg1 = "It has been now created with also empty metadata files to be filled for each station."
        raise ValueError(msg + msg1)

    # -------------------------------------------------------------------------.
    # Check there are metadata file for each station_id in /metadata
    list_metadata_fpath = glob.glob(os.path.join(metadata_dir, "*.yml"))
    list_metadata_fname = [os.path.basename(fpath) for fpath in list_metadata_fpath]
    list_metadata_station_id = [fname.strip(".yml") for fname in list_metadata_fname]
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
    # Check metadata compliance
    _ = [check_metadata_compliance(fpath) for fpath in list_metadata_fpath]
    # -------------------------------------------------------------------------.


def check_processed_dir(processed_dir, force=False):
    """Check that 'processed_dir' is a valid directory path."""
    if not isinstance(processed_dir, str):
        raise TypeError("Provide 'processed_dir' as a string'.")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if not force:
        if os.path.exists(processed_dir):
            raise ValueError(
                "'processed_dir' {} already exists and force=False.".format(
                    processed_dir
                )
            )
    else:
        if os.path.exists(processed_dir):
            if not os.path.isdir(processed_dir):
                raise ValueError(
                    "'processed_dir' {} already exist but is not a directory.".format(
                        processed_dir
                    )
                )


def check_metadata_dir(processed_path):
    # Create metadata folder
    try:
        metadata_folder_path = os.path.join(processed_path, "metadata")
        os.makedirs(metadata_folder_path)
        logger.debug(f"Created {metadata_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {metadata_folder_path}")
        pass
    except (Exception) as e:
        msg = (
            f"Can not create folder metadata inside <metadata_folder_path>. Error: {e}"
        )
        logger.exception(msg)
        raise FileNotFoundError(msg)


def check_campaign_name(raw_dir, processed_dir):
    """Check that 'raw_dir' and 'processed_dir' have same campaign_name."""
    upper_campaign_name = os.path.basename(raw_dir).upper()
    raw_campaign_name = os.path.basename(raw_dir)
    processed_campaign_name = os.path.basename(processed_dir)
    if raw_campaign_name != processed_campaign_name:
        raise ValueError(
            f"'raw_dir' and 'processed_dir' must ends with same <campaign_name> {upper_campaign_name}"
        )
    if raw_campaign_name != upper_campaign_name:
        raise ValueError(
            f"'raw_dir' and 'processed_dir' must ends with UPPERCASE <campaign_name> {upper_campaign_name}"
        )
    return upper_campaign_name


def check_directories(raw_dir, processed_dir, force=False):
    """Check that the specified directories respect the standards."""
    raw_dir = parse_fpath(raw_dir)
    processed_dir = parse_fpath(processed_dir)
    check_raw_dir(raw_dir)
    check_processed_dir(processed_dir, force=force)
    check_campaign_name(raw_dir, processed_dir)
    return raw_dir, processed_dir


####--------------------------------------------------------------------------.
#### Directory structure


def create_directory_structure(raw_dir, processed_dir):
    """Create directory structure for L0 and L1 processing."""
    campaign_name = os.path.basename(raw_dir).upper()

    # Creation metadata folder inside campaing processed folder
    check_metadata_dir(processed_dir)

    # Create info folder inside  processed_dir
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

    # Create info L0 folder inside  processed_dir
    try:
        L0_folder_path = os.path.join(processed_dir, "L0")
        os.makedirs(L0_folder_path)
        logger.debug(f"Created {L0_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {L0_folder_path}")
        pass
    except (Exception) as e:
        msg = f"Can not create folder L0 inside <station_folder_path>. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)

    # Create info L1 folder inside  processed_dir
    try:
        L1_folder_path = os.path.join(processed_dir, "L1")
        os.makedirs(L1_folder_path)
        logger.debug(f"Created {L1_folder_path}")
    except FileExistsError:
        logger.debug(f"Found {L1_folder_path}")
        pass
    except (Exception) as e:
        msg = f"Can not create folder L0 1inside <L1_folder_path>. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)

    return


####--------------------------------------------------------------------------.
#### DISDRODB Readers


def check_L0_is_available(processed_dir, station_id):
    fpath = get_L0_fpath(processed_dir, station_id)
    if not os.path.exists(fpath):
        msg = "Need to run L0 processing first. The Apache Parquet file {fpath} is not available."
        logger.exception(msg)
        raise ValueError(msg)
    # Log
    msg = f"Found parquet file: {fpath}"
    logger.info(msg)


def read_L0_data(
    processed_dir, station_id, lazy=True, verbose=False, debugging_mode=False
):
    """Read L0 Apache Parquet into dataframe.

    If debugging_mode = True, return just a subset of total rows.
    """
    # Check L0 is available
    check_L0_is_available(processed_dir, station_id)
    # Define fpath
    fpath = get_L0_fpath(processed_dir, station_id)
    # Log
    msg = f" - Reading L0 Apache Parquet file at {fpath} started"
    if verbose:
        print(msg)
    logger.info(msg)
    # Read
    if lazy:
        df = dd.read_parquet(fpath)
    else:
        df = pd.read_parquet(fpath)
    # Log
    msg = f" - Reading L0 Apache Parquet file at {fpath} ended"
    if verbose:
        print(msg)
    logger.info(msg)
    # Subset dataframe if debugging_mode = True
    if debugging_mode:
        if not lazy:
            df = df.iloc[0:100, :]
        else:
            NotImplementedError
    return df


####--------------------------------------------------------------------------.
#### TODO: include in create_directory_structure

####--------------------------------------------------------------------------.
