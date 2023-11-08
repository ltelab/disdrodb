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
    create_required_directory,
)
from disdrodb.utils.logger import log_info

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
    from disdrodb.l0.check_campaign_directories import check_processed_dir, check_raw_dir

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
    from disdrodb.l0.check_campaign_directories import check_processed_dir

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
