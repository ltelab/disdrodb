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
"""Routines tot extract information from the DISDRODB infrastructure."""

import os

import numpy as np

from disdrodb.api.checks import check_product
from disdrodb.configs import get_base_dir
from disdrodb.utils.directories import check_directory_exists, count_files, list_directories, list_files


def get_disdrodb_path(
    base_dir,
    product,
    data_source="",
    campaign_name="",
    check_exists=True,
):
    """Return the directory in the DISDRODB infrastructure.

    If data_source and campaign_name are not specified it return the product directory.
    If data_source is specified, it returns the data_source directory.
    If campaign_source is specified, it returns the campaign_name directory.

    Parameters
    ----------
    base_dir : str
        The disdrodb base directory
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str, optional
        The data source. Must be specified if campaign_name is specified.
    campaign_name : str, optional
        The campaign_name.
    check_exists : bool, optional
        Whether to check if the directory exists. By default True.

    Returns
    -------
    dir_path : str
        Directory path
    """
    from disdrodb.api.checks import check_base_dir
    from disdrodb.utils.directories import check_directory_exists

    # Check base_dir validity
    base_dir = check_base_dir(base_dir)
    if len(campaign_name) > 0:
        if len(data_source) == 0:
            raise ValueError("If campaign_name is specified, data_source must be specified.")

    # Get directory
    if product.upper() == "RAW":
        dir_path = os.path.join(base_dir, "Raw", data_source, campaign_name)
    else:
        dir_path = os.path.join(base_dir, "Processed", data_source, campaign_name)
    if check_exists:
        check_directory_exists(dir_path)
    return dir_path


def define_campaign_dir(
    product,
    data_source,
    campaign_name,
    base_dir=None,
    check_exists=False,
):
    """Return the campaign directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str
        The data source. Must be specified if campaign_name is specified.
    campaign_name : str
        The campaign_name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    campaign_dir = get_disdrodb_path(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    return str(campaign_dir)


def define_station_dir(
    product,
    data_source,
    campaign_name,
    station_name,
    base_dir=None,
    check_exists=False,
):
    """Return the station data directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    campaign_dir = get_disdrodb_path(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    if product.upper() == "RAW":
        station_dir = os.path.join(campaign_dir, "data", station_name)
    else:
        station_dir = os.path.join(campaign_dir, product, station_name)
    if check_exists:
        check_directory_exists(station_dir)
    return str(station_dir)


def define_metadata_dir(
    product,
    data_source,
    campaign_name,
    base_dir=None,
    check_exists=False,
):
    """Return the metadata directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    metadata_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    campaign_dir = define_campaign_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    if product.upper() == "RAW":
        metadata_dir = os.path.join(campaign_dir, "metadata")
    else:
        metadata_dir = os.path.join(campaign_dir, "metadata")
    if check_exists:
        check_directory_exists(metadata_dir)
    return str(metadata_dir)


def define_metadata_filepath(
    product,
    data_source,
    campaign_name,
    station_name,
    base_dir=None,
    check_exists=False,
):
    """Return the station metadata filepath in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    metadata_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    metadata_dir = define_metadata_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    metadata_fpath = os.path.join(metadata_dir, f"{station_name}.yml")

    if check_exists and not os.path.exists(metadata_fpath):
        raise ValueError(f"The metadata file for {station_name} at {metadata_fpath} does not exists.")

    return str(metadata_fpath)


def define_config_dir(product):
    """Define the config directory path of a given DISDRODB product."""
    from disdrodb import __root_path__

    if product.upper() in ["RAW", "L0A", "L0B"]:
        dir_name = "l0"
    else:
        raise NotImplementedError(f"Product {product} not implemented.")
    config_dir_path = os.path.join(__root_path__, "disdrodb", dir_name, "configs")
    return config_dir_path


####---------------------------------------------------------------------------.


def _get_list_stations_dirs(product, campaign_dir):
    # Get directory where data are stored
    # - Raw: <campaign>/data/<...>
    # - Processed: <campaign>/L0A/L0B>
    if product.upper() == "RAW":
        data_path = os.path.join(campaign_dir, "data")
    else:
        data_path = os.path.join(campaign_dir, product)
    # Check if the data directory exists
    # - For a fresh disdrodb-data cloned repo, no "data" directories
    if not os.path.exists(data_path):
        return []
    # Get list of directories (stations)
    list_stations = os.listdir(data_path)
    list_stations_dir = [os.path.join(data_path, station_name) for station_name in list_stations]
    return list_stations_dir


def _get_list_stations_with_data(product, campaign_dir):
    """Get the list of stations with data inside."""
    # Get stations directory
    list_stations_dir = _get_list_stations_dirs(product=product, campaign_dir=campaign_dir)
    # Count number of files within directory
    list_nfiles_per_station = [count_files(station_dir, "*", recursive=True) for station_dir in list_stations_dir]
    # Keep only stations with at least one file
    list_stations = [os.path.basename(path) for n, path in zip(list_nfiles_per_station, list_stations_dir) if n >= 1]
    return list_stations


def _get_list_stations_with_metadata(campaign_dir):
    # Get directory where metadata are stored
    metadata_path = os.path.join(campaign_dir, "metadata")
    # List metadata files
    metadata_filepaths = list_files(metadata_path, glob_pattern="*.yml", recursive=False)
    # Return stations with metadata
    list_stations = [os.path.basename(fpath).replace(".yml", "") for fpath in metadata_filepaths]
    return list_stations


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
    list_stations = list(set(list_stations_data).intersection(list_stations_metadata))

    # Return all available stations for a give campaign
    return list_stations


def _get_campaigns_stations(base_dir, product, data_source, campaign_names):
    if isinstance(campaign_names, str):
        campaign_names = [campaign_names]
    list_available_stations = []
    for campaign_name in campaign_names:
        # Get list of available stations
        list_stations = _get_campaign_stations(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
        )
        for station_name in list_stations:
            list_available_stations.append((data_source, campaign_name, station_name))

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
    list_available_stations = []
    for data_source in data_sources:
        list_available = _get_data_source_stations(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
        )
        for station_name in list_available:
            list_available_stations.append(station_name)

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
        raise ValueError(f"These data sources are invalid: {invalid_data_sources}.")
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
        raise ValueError(f"These campaign names are invalid: {invalid_campaign_names}.")
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
    else:
        data_source_campaigns = [(info[0], info[1]) for info in list_available_stations]
        data_source_campaigns = list(set(data_source_campaigns))
        return data_source_campaigns


def available_stations(
    product,
    data_sources=None,
    campaign_names=None,
    return_tuple=True,
    base_dir=None,
):
    """Return stations for which data are available."""
    base_dir = get_base_dir(base_dir)
    # Checks
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
    # Format arguments to list
    if isinstance(data_sources, str):
        data_sources = [data_sources]
    if isinstance(campaign_names, str):
        campaign_names = [campaign_names]

    # If data_source is None, first retrieve all stations
    if data_sources is None:
        list_info = _get_stations(base_dir=base_dir, product=product)
    # Otherwise retrieve all stations for the specified data sources
    else:
        list_info = _get_data_sources_stations(
            base_dir=base_dir,
            data_sources=data_sources,
            product=product,
        )

    # Then, if campaign_name is not None, subset by campaign_name
    if campaign_names is not None:
        list_info = [info for info in list_info if info[1] in campaign_names]

    if return_tuple:
        return list_info
    else:
        # TODO: ENSURE THAT NO DUPLICATED STATION NAMES ?
        list_stations = [info[2] for info in list_info]
        return list_stations
