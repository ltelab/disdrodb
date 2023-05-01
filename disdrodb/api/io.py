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

import os
import glob
import yaml
import numpy as np


def _get_disdrodb_directory(
    disdrodb_dir,
    product_level,
    data_source="",
    campaign_name="",
    check_exist=True,
):
    from disdrodb.l0.io import _check_directory_exist
    from disdrodb.api.checks import check_disdrodb_dir

    # Check if disdrodb_dir has a valid name
    check_disdrodb_dir(disdrodb_dir)

    # Get directory
    if product_level.upper() == "RAW":
        dir_path = os.path.join(disdrodb_dir, "Raw", data_source, campaign_name)
    else:
        dir_path = os.path.join(disdrodb_dir, "Processed", data_source, campaign_name)
    if check_exist:
        _check_directory_exist(dir_path)
    return dir_path


def _get_list_stations_dirs(product_level, campaign_dir):
    # Get directory where data are stored
    # - Raw: <campaign>/data/<...>
    # - Processed: <campaign>/L0A/L0B>
    if product_level.upper() == "RAW":
        data_path = os.path.join(campaign_dir, "data")
    else:
        data_path = os.path.join(campaign_dir, product_level)
    # Get list of directories (stations)
    list_stations = os.listdir(data_path)
    list_stations_dir = [os.path.join(data_path, station_name) for station_name in list_stations]
    return list_stations_dir


def _get_list_stations_with_data(product_level, campaign_dir):
    """Get the list of stations with data inside."""
    # Get stations directory
    list_stations_dir = _get_list_stations_dirs(product_level=product_level, campaign_dir=campaign_dir)
    # Count number of files within directory
    list_nfiles_per_station = [len(glob.glob(os.path.join(path, "*"))) for path in list_stations_dir]
    # Keep only stations with at least one file
    list_stations = [os.path.basename(path) for n, path in zip(list_nfiles_per_station, list_stations_dir) if n >= 1]
    return list_stations


def _get_list_stations_with_metadata(campaign_dir):
    # Get directory where metadata are stored
    metadata_path = os.path.join(campaign_dir, "metadata")
    # List metadata files
    list_metadata_files = glob.glob(os.path.join(metadata_path, "*.yml"))
    # Return stations with metadata
    list_stations = [os.path.basename(fpath).replace(".yml", "") for fpath in list_metadata_files]
    return list_stations


def _get_campaign_stations(disdrodb_dir, product_level, data_source, campaign_name):
    # Get campaign directory
    campaign_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_source=data_source,
        campaign_name=campaign_name,
    )

    # Get list of stations with data and metadata
    list_stations_data = _get_list_stations_with_data(product_level=product_level, campaign_dir=campaign_dir)
    list_stations_metadata = _get_list_stations_with_metadata(campaign_dir)
    # Get list of stations with both data and metadata
    list_stations = list(set(list_stations_data).intersection(list_stations_metadata))

    # Return all available stations for a give campaign
    return list_stations


def _get_campaigns_stations(disdrodb_dir, product_level, data_source, campaign_names):
    if isinstance(campaign_names, str):
        campaign_names = [campaign_names]
    list_available_stations = []
    for campaign_name in campaign_names:
        # Get list of available stations
        list_stations = _get_campaign_stations(
            disdrodb_dir=disdrodb_dir,
            product_level=product_level,
            data_source=data_source,
            campaign_name=campaign_name,
        )
        for station_name in list_stations:
            list_available_stations.append((data_source, campaign_name, station_name))

    # Return all available stations for the asked campaigns (and specific data source)
    return list_available_stations


def _get_data_source_stations(disdrodb_dir, product_level, data_source):
    """Return list of available stations for a specific data source.

    Returns a tuple (<data_source>, <campaign_name>, <station_name>)
    """
    # Get data source directory
    data_source_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_source=data_source,
        campaign_name="",
    )
    # Get possible campaign list
    campaign_names = os.listdir(data_source_dir)

    # For each campaign, retrieve available stations
    list_available_stations = _get_campaigns_stations(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_source=data_source,
        campaign_names=campaign_names,
    )

    # Return all available stations for a specific data source
    return list_available_stations


def _get_data_sources_stations(disdrodb_dir, product_level, data_sources):
    if isinstance(data_sources, str):
        data_sources = [data_sources]
    list_available_stations = []
    for data_source in data_sources:
        list_available = _get_data_source_stations(
            disdrodb_dir=disdrodb_dir,
            product_level=product_level,
            data_source=data_source,
        )
        for station_name in list_available:
            list_available_stations.append(station_name)

    # Return all available stations
    return list_available_stations


def _get_stations(disdrodb_dir, product_level):
    # Get raw or processed directory
    level_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_source="",
        campaign_name="",
    )
    # Get possible data sources
    data_sources = os.listdir(level_dir)

    # For each data_source, retrieve available stations
    list_available_stations = _get_data_sources_stations(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_sources=data_sources,
    )
    # Return all available stations
    return list_available_stations


def _get_metadata_fpath(disdrodb_dir, product_level, data_source, campaign_name, station_name):
    """Get metadata file path a given station."""
    campaign_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    metadata_fpath = os.path.join(campaign_dir, "metadata", station_name + ".yml")
    return metadata_fpath


def get_metadata_dict(disdrodb_dir, product_level, data_source, campaign_name, station_name):
    """Get metadata of a given station."""
    metadata_fpath = _get_metadata_fpath(disdrodb_dir, product_level, data_source, campaign_name, station_name)
    with open(metadata_fpath, "r") as f:
        metadata_dict = yaml.safe_load(f)
    return metadata_dict


####---------------------------------------------------------------------------.
#### I/O CHECKS
def check_product_level(product_level):
    """Check DISDRODB product level."""
    if not isinstance(product_level, str):
        raise TypeError("`product_level` must be a string.")
    valid_product_levels = ["RAW", "L0A", "L0B"]
    if product_level.upper() not in valid_product_levels:
        raise ValueError(f"Valid `product_levels` are {valid_product_levels}.")
    return product_level


def check_data_sources(disdrodb_dir, product_level, data_sources):
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
    dir_path = _get_disdrodb_directory(disdrodb_dir=disdrodb_dir, product_level=product_level)
    # Get data sources directory
    list_dir = os.listdir(dir_path)
    # Check if there are unvalid data_sources
    idx_unvalid = np.where(np.isin(data_sources, list_dir, invert=True))[0]
    if len(idx_unvalid) > 0:
        unvalid_data_sources = data_sources[idx_unvalid].tolist()
        raise ValueError(f"These data sources are unvalid: {unvalid_data_sources}.")
    # Return data_sources list
    data_sources = data_sources.tolist()
    return data_sources


def _check_campaign_names(disdrodb_dir, product_level, campaign_names):
    """Check DISDRODB campaign_names are valid.

    It checks only if the directory exist within the product level.
    """
    # If campaign_names is None, return None
    if isinstance(campaign_names, type(None)):
        return campaign_names
    # Ensure is a list
    if isinstance(campaign_names, str):
        campaign_names = [campaign_names]
    # Remove duplicates
    campaign_names = np.unique(np.array(campaign_names))
    # Get product level directory path
    dir_path = _get_disdrodb_directory(disdrodb_dir=disdrodb_dir, product_level=product_level)
    # Get campaigns directory path
    list_campaigns_path = glob.glob(os.path.join(dir_path, "*", "*"))
    # Get campaigns names
    list_campaign_names = [os.path.basename(path) for path in list_campaigns_path]
    # Remove duplicates
    list_campaign_names = np.unique(list_campaign_names)
    # Check if there are unvalid campaign_names
    idx_unvalid = np.where(np.isin(campaign_names, list_campaign_names, invert=True))[0]
    if len(idx_unvalid) > 0:
        unvalid_campaign_names = campaign_names[idx_unvalid].tolist()
        raise ValueError(f"These campaign names are unvalid: {unvalid_campaign_names}.")
    # Return campaign_names list
    campaign_names = campaign_names.tolist()
    return campaign_names


####---------------------------------------------------------------------------.
#### DISDRODB I/O INTERFACE


def available_data_sources(disdrodb_dir, product_level):
    """Return data sources for which stations data are available."""
    product_level = check_product_level(product_level)
    # Get available stations
    list_available_stations = _get_stations(disdrodb_dir=disdrodb_dir, product_level=product_level)
    data_sources = [info[0] for info in list_available_stations]
    data_sources = np.unique(data_sources).tolist()
    return data_sources


def available_campaigns(disdrodb_dir, product_level, data_sources=None, return_tuple=True):
    """Return campaigns for which stations data are available."""
    # Checks
    product_level = check_product_level(product_level)
    data_sources = check_data_sources(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_sources=data_sources,
    )
    # Get available stations
    if data_sources is None:
        list_available_stations = _get_stations(disdrodb_dir=disdrodb_dir, product_level=product_level)
    else:
        list_available_stations = _get_data_sources_stations(
            disdrodb_dir=disdrodb_dir,
            product_level=product_level,
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
    disdrodb_dir,
    product_level,
    data_sources=None,
    campaign_names=None,
    return_tuple=True,
):
    """Return stations for which data are available."""
    # Checks
    product_level = check_product_level(product_level)
    data_sources = check_data_sources(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_sources=data_sources,
    )
    campaign_names = _check_campaign_names(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        campaign_names=campaign_names,
    )
    # Format arguments to list
    if isinstance(data_sources, str):
        data_sources = [data_sources]
    if isinstance(campaign_names, str):
        campaign_names = [campaign_names]

    # If data_source is None, first retrieve all stations
    if data_sources is None:
        list_info = _get_stations(disdrodb_dir=disdrodb_dir, product_level=product_level)
    # Otherwise retrieve all stations for the specified data sources
    else:
        list_info = _get_data_sources_stations(
            disdrodb_dir=disdrodb_dir,
            data_sources=data_sources,
            product_level=product_level,
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
