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
"""Routines to list and open DISDRODB products."""

import os
import shutil
from typing import Optional

import numpy as np
import xarray as xr

from disdrodb.api.checks import check_product
from disdrodb.api.path import define_data_dir, define_product_dir, get_disdrodb_path
from disdrodb.configs import get_base_dir
from disdrodb.utils.directories import count_files, list_directories, list_files
from disdrodb.utils.logger import (
    log_info,
)


def get_required_product(product):
    """Determine the required product for input product processing."""
    # Check input
    check_product(product)
    # Determine required product
    requirement_dict = {
        "L0A": "RAW",
        "L0B": "L0A",
        "L0C": "L0B",
        "L1": "L0C",
        "L2E": "L1",
        "L2M": "L2E",
    }
    required_product = requirement_dict[product]
    return required_product


def filter_filepaths(filepaths, debugging_mode):
    """Filter out filepaths if ``debugging_mode=True``."""
    if debugging_mode:
        max_files = min(3, len(filepaths))
        filepaths = filepaths[0:max_files]
    return filepaths


def find_files(
    data_source,
    campaign_name,
    station_name,
    product,
    model_name=None,
    sample_interval=None,
    rolling=None,
    debugging_mode: bool = False,
    base_dir: Optional[str] = None,
):
    """Retrieve DISDRODB product files for a give station.

    Parameters
    ----------
    data_source : str
        The name of the institution (for campaigns spanning multiple countries) or
        the name of the country (for campaigns or sensor networks within a single country).
        Must be provided in UPPER CASE.
    campaign_name : str
        The name of the campaign. Must be provided in UPPER CASE.
    station_name : str
        The name of the station.
    product : str
        The name DISDRODB product.
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    model_name : str
        The model name of the statistical distribution for the DSD.
        It must be specified only for product L2M !
    debugging_mode : bool, optional
        If ``True``, it select maximum 3 files for debugging purposes.
        The default is ``False``.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    Returns
    -------
    filepaths : list
        List of file paths.

    """
    # Retrieve data directory
    data_dir = define_data_dir(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        # Option for L2E and L2M
        sample_interval=sample_interval,
        rolling=rolling,
        # Options for L2M
        model_name=model_name,
    )

    # Define glob pattern
    glob_pattern = "*.parquet" if product == "L0A" else "*.nc"

    # Retrieve files
    filepaths = list_files(data_dir, glob_pattern=glob_pattern, recursive=True)

    # Filter out filepaths if debugging_mode=True
    filepaths = filter_filepaths(filepaths, debugging_mode=debugging_mode)

    # If no file available, raise error
    if len(filepaths) == 0:
        msg = f"No {product} files are available in {data_dir}. Run {product} processing first."
        raise ValueError(msg)

    # Sort filepaths
    filepaths = sorted(filepaths)

    return filepaths


def open_dataset(
    data_source,
    campaign_name,
    station_name,
    product,
    model_name=None,
    sample_interval=None,
    rolling=None,
    debugging_mode: bool = False,
    base_dir: Optional[str] = None,
    **open_kwargs,
):
    """Retrieve DISDRODB product files for a give station.

    Parameters
    ----------
    data_source : str
        The name of the institution (for campaigns spanning multiple countries) or
        the name of the country (for campaigns or sensor networks within a single country).
        Must be provided in UPPER CASE.
    campaign_name : str
        The name of the campaign. Must be provided in UPPER CASE.
    station_name : str
        The name of the station.
    product : str
        The name DISDRODB product.
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    model_name : str
        The model name of the statistical distribution for the DSD.
        It must be specified only for product L2M !
    debugging_mode : bool, optional
        If ``True``, it select maximum 3 files for debugging purposes.
        The default is ``False``.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    Returns
    -------
    xr.Dataset

    """
    # Check product validity
    if product == "RAW":
        raise ValueError("It's not possible to open the raw data with this function.")
    # List product files
    filepaths = find_files(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        debugging_mode=debugging_mode,
        model_name=model_name,
        sample_interval=sample_interval,
        rolling=rolling,
    )
    # Open L0A Parquet files
    if product == "L0A":
        # TODO: with pandas?
        raise NotImplementedError

    # Open DISDRODB netCDF files using xarray
    # - TODO: parallel option and add closers !
    # - decode_timedelta -- > sample_interval not decoded to timedelta !
    list_ds = [xr.open_dataset(fpath, decode_timedelta=False, **open_kwargs) for fpath in filepaths]
    ds = xr.concat(list_ds, dim="time")
    return ds


def _get_list_stations_dirs(product, campaign_dir):
    # Get directory where data are stored
    product_dir = define_product_dir(campaign_dir=campaign_dir, product=product)
    # Check if the data directory exists
    # - For a fresh disdrodb-data cloned repo, no "data" directories
    if not os.path.exists(product_dir):
        return []
    # Get list of directories (stations)
    stations_names = os.listdir(product_dir)
    list_stations_dir = [os.path.join(product_dir, station_name) for station_name in stations_names]
    return list_stations_dir


def _get_list_stations_with_data(product, campaign_dir):
    """Get the list of stations with data inside."""
    # Get stations directory
    list_stations_dir = _get_list_stations_dirs(product=product, campaign_dir=campaign_dir)
    # Count number of files within directory
    # - TODO: here just check for one file !
    list_nfiles_per_station = [count_files(station_dir, "*", recursive=True) for station_dir in list_stations_dir]
    # Keep only stations with at least one file
    stations_names = [os.path.basename(path) for n, path in zip(list_nfiles_per_station, list_stations_dir) if n >= 1]
    return stations_names


def _get_list_stations_with_metadata(campaign_dir):
    # Get directory where metadata are stored
    metadata_path = os.path.join(campaign_dir, "metadata")
    # List metadata files
    metadata_filepaths = list_files(metadata_path, glob_pattern="*.yml", recursive=False)
    # Return stations with metadata
    stations_names = [
        os.path.basename(filepath).replace(".yml", "") for filepath in metadata_filepaths
    ]  # TODO: use api.path
    return stations_names


def _get_campaign_stations(base_dir, product, data_source, campaign_name):
    # Get campaign directory
    campaign_dir = get_disdrodb_path(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    # Get list of stations with data and metadata
    list_stations_data = _get_list_stations_with_data(product=product, campaign_dir=campaign_dir)
    list_stations_metadata = _get_list_stations_with_metadata(campaign_dir)
    # Get list of stations with both data and metadata
    stations_names = list(set(list_stations_data).intersection(list_stations_metadata))

    # Return all available stations for a give campaign
    return stations_names


def _get_campaigns_stations(base_dir, product, data_source, campaign_names):
    if isinstance(campaign_names, str):
        campaign_names = [campaign_names]
    list_available_stations = [
        (data_source, campaign_name, station_name)
        for campaign_name in campaign_names
        for station_name in _get_campaign_stations(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
        )
    ]
    # Return all available stations for the asked campaigns (and specific data source)
    return list_available_stations


def _get_data_source_stations(base_dir, product, data_source):
    """Return list of available stations for a specific data source.

    Returns a tuple (<DATA_SOURCE>, <CAMPAIGN_NAME>, <station_name>)
    """
    # Get data source directory
    data_source_dir = get_disdrodb_path(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name="",
    )
    # Get possible campaign list
    campaign_names = os.listdir(data_source_dir)

    # For each campaign, retrieve available stations
    list_available_stations = _get_campaigns_stations(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_names=campaign_names,
    )

    # Return all available stations for a specific data source
    return list_available_stations


def _get_data_sources_stations(base_dir, product, data_sources):
    if isinstance(data_sources, str):
        data_sources = [data_sources]
    list_available_stations = [
        station_name
        for data_source in data_sources
        for station_name in _get_data_source_stations(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
        )
    ]
    # Return all available stations
    return list_available_stations


def _get_stations(base_dir, product):
    # Get raw or processed directory
    level_dir = get_disdrodb_path(
        base_dir=base_dir,
        product=product,
        data_source="",
        campaign_name="",
    )
    # Get possible data sources
    data_sources = os.listdir(level_dir)

    # For each data_source, retrieve available stations
    list_available_stations = _get_data_sources_stations(
        base_dir=base_dir,
        product=product,
        data_sources=data_sources,
    )
    # Return all available stations
    return list_available_stations


####---------------------------------------------------------------------------.
#### I/O CHECKS


def _check_data_sources(base_dir, product, data_sources):
    """Check DISDRODB data source.

    It checks only if the directory exist.
    """
    # If data_sources is None, return None
    if isinstance(data_sources, type(None)):
        return data_sources
    # Ensure is a list
    if isinstance(data_sources, str):
        data_sources = [data_sources]
    # Remove duplicates
    data_sources = np.unique(np.array(data_sources))
    # Get directory
    dir_path = get_disdrodb_path(base_dir=base_dir, product=product)
    # Get data sources directory
    list_dir = os.listdir(dir_path)
    # Check if there are invalid data_sources
    idx_invalid = np.where(np.isin(data_sources, list_dir, invert=True))[0]
    if len(idx_invalid) > 0:
        invalid_data_sources = data_sources[idx_invalid].tolist()
        raise ValueError(f"These data sources does not exist: {invalid_data_sources}.")
    # Return data_sources list
    data_sources = data_sources.tolist()
    return data_sources


def _check_campaign_names(base_dir, product, campaign_names):
    """Check DISDRODB campaign_names are valid.

    It checks only if the directory exist within the product.
    """
    # If campaign_names is None, return None
    if isinstance(campaign_names, type(None)):
        return campaign_names
    # Ensure is a list
    if isinstance(campaign_names, str):
        campaign_names = [campaign_names]
    # Remove duplicates
    campaign_names = np.unique(np.array(campaign_names))
    # Get product directory path
    dir_path = get_disdrodb_path(base_dir=base_dir, product=product)
    # Get campaigns directory path
    list_campaigns_path = list_directories(dir_path, glob_pattern=os.path.join("*", "*"), recursive=False)
    # Get campaigns names
    list_campaign_names = [os.path.basename(path) for path in list_campaigns_path]
    # Remove duplicates
    list_campaign_names = np.unique(list_campaign_names)
    # Check if there are invalid campaign_names
    idx_invalid = np.where(np.isin(campaign_names, list_campaign_names, invert=True))[0]
    if len(idx_invalid) > 0:
        invalid_campaign_names = campaign_names[idx_invalid].tolist()
        raise ValueError(f"These campaign names does not exist: {invalid_campaign_names}.")
    # Return campaign_names list
    campaign_names = campaign_names.tolist()
    return campaign_names


####---------------------------------------------------------------------------.
#### DISDRODB I/O INTERFACE


def available_data_sources(product, base_dir=None):
    """Return data sources for which stations data are available."""
    base_dir = get_base_dir(base_dir)
    product = check_product(product)
    # Get available stations
    list_available_stations = _get_stations(base_dir=base_dir, product=product)
    data_sources = [info[0] for info in list_available_stations]
    data_sources = np.unique(data_sources).tolist()
    return data_sources


def available_campaigns(product, data_sources=None, return_tuple=True, base_dir=None):
    """Return campaigns for which stations data are available."""
    # Checks
    base_dir = get_base_dir(base_dir)
    product = check_product(product)
    data_sources = _check_data_sources(
        base_dir=base_dir,
        product=product,
        data_sources=data_sources,
    )
    # Get available stations
    if data_sources is None:
        list_available_stations = _get_stations(base_dir=base_dir, product=product)
    else:
        list_available_stations = _get_data_sources_stations(
            base_dir=base_dir,
            product=product,
            data_sources=data_sources,
        )
    if not return_tuple:
        campaigns = [info[1] for info in list_available_stations]
        campaigns = np.unique(campaigns).tolist()
        return campaigns
    data_source_campaigns = [(info[0], info[1]) for info in list_available_stations]
    data_source_campaigns = list(set(data_source_campaigns))
    return data_source_campaigns


def available_stations(
    product,
    data_sources=None,
    campaign_names=None,
    station_names=None,
    return_tuple=True,
    raise_error_if_empty=False,
    base_dir=None,
):
    """Return stations for which data and metadata are available on disk.

    Raise an error if no stations are available.
    """
    base_dir = get_base_dir(base_dir)

    # Checks arguments
    product = check_product(product)
    data_sources = _check_data_sources(
        base_dir=base_dir,
        product=product,
        data_sources=data_sources,
    )
    campaign_names = _check_campaign_names(
        base_dir=base_dir,
        product=product,
        campaign_names=campaign_names,
    )
    if isinstance(station_names, str):
        station_names = [station_names]

    # If data_source is None, retrieve all stations
    if data_sources is None:
        list_info = _get_stations(base_dir=base_dir, product=product)

    ###-----------------------------------------------.
    ### Filter by data_sources
    else:
        list_info = _get_data_sources_stations(
            base_dir=base_dir,
            data_sources=data_sources,
            product=product,
        )
    # If no stations available, raise an error
    if raise_error_if_empty and len(list_info) == 0:
        raise ValueError(f"No stations available given the provided `data_sources` {data_sources}.")

    ###-----------------------------------------------.
    ### Filter by campaign_names
    # If campaign_names is not None, subset by campaign_names
    if campaign_names is not None:
        list_info = [info for info in list_info if info[1] in campaign_names]

    # If no stations available, raise an error
    if raise_error_if_empty and len(list_info) == 0:
        raise ValueError(f"No stations available given the provided `campaign_names` {campaign_names}.")

    ###-----------------------------------------------.
    ### Filter by station_names
    # If station_names is not None, subset by station_names
    if station_names is not None:
        list_info = [info for info in list_info if info[2] in station_names]
    # If no stations available, raise an error
    if raise_error_if_empty and len(list_info) == 0:
        raise ValueError(f"No stations available given the provided `station_names` {station_names}.")

    ###-----------------------------------------------.
    # Return list with the tuple (data_source, campaign_name, station_name)
    if return_tuple:
        return list_info

    # - Return list with the name of the available stations
    list_stations = [info[2] for info in list_info]
    return list_stations


####----------------------------------------------------------------------------------
#### DISDRODB Removal Functions


def remove_product(
    base_dir,
    product,
    data_source,
    campaign_name,
    station_name,
    logger=None,
    verbose=True,
):
    """Remove all product files of a specific station."""
    if product.upper() == "RAW":
        raise ValueError("Removal of 'RAW' files is not allowed.")
    data_dir = define_data_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    if logger is not None:
        log_info(logger=logger, msg="Removal of {product} files started.", verbose=verbose)
    shutil.rmtree(data_dir)
    if logger is not None:
        log_info(logger=logger, msg="Removal of {product} files ended.", verbose=verbose)
