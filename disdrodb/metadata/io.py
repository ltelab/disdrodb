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
"""Routines to manipulate the DISDRODB Metadata Archive."""

import glob
import os

from disdrodb.api.info import infer_campaign_name_from_path, infer_data_source_from_path
from disdrodb.api.path import define_metadata_filepath
from disdrodb.configs import get_base_dir
from disdrodb.metadata.manipulation import sort_metadata_dictionary
from disdrodb.metadata.standards import get_valid_metadata_keys
from disdrodb.utils.yaml import read_yaml, write_yaml


def read_station_metadata(data_source, campaign_name, station_name, base_dir=None, product="RAW"):
    """Open the station metadata YAML file into a dictionary.

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
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    product : str, optional
        The DISDRODB product in which to search for the metadata file.
        The default is "RAW".

    Returns
    -------
    metadata: dictionary
        The station metadata dictionary

    """
    # Retrieve metadata filepath
    metadata_filepath = define_metadata_filepath(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        check_exists=True,
    )
    metadata_dict = read_yaml(metadata_filepath)
    return metadata_dict


def get_list_metadata(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    with_stations_data=True,
    base_dir=None,
):
    """
    Get the list of metadata filepaths in the DISDRODB raw archive.

    Parameters
    ----------
    data_sources : str or list of str
        Name of data source(s) of interest.
        The name(s) must be UPPER CASE.
        The default is None
    campaign_names : str or list of str
        Name of the campaign(s) of interest.
        The name(s) must be UPPER CASE.
        The default is None
    station_names : str or list of str
        Station names of interest.
        The default is None
    with_stations_data : bool
        If True, only return metadata filepaths that have corresponding data in the local DISDRODB raw archive.
        The default is True
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    metadata_filepaths: list
        List of metadata YAML file paths

    """
    base_dir = get_base_dir(base_dir)
    if with_stations_data:
        list_metadata = _get_list_metadata_with_data(
            base_dir=base_dir,
            data_sources=data_sources,
            campaign_names=campaign_names,
            station_names=station_names,
        )
    else:
        list_metadata = _get_list_all_metadata(
            base_dir=base_dir,
            data_sources=data_sources,
            campaign_names=campaign_names,
            station_names=station_names,
        )
    return list_metadata


def _get_list_all_metadata(base_dir, data_sources=None, campaign_names=None, station_names=None):
    """
    Get the list of metadata filepaths in the DISDRODB raw archive.

    Parameters
    ----------
    base_dir : str
        Base directory of DISDRODB
        Format: <...>/DISDRODB
    data_sources : str or list of str
        Name of data source(s) of interest.
        The name(s) must be UPPER CASE.
        The default is None
    campaign_names : str or list of str
        Name of the campaign(s) of interest.
        The name(s) must be UPPER CASE.
        The default is None
    station_names : str or list of str
        Station names of interest.
        The default is None
    with_stations_data : bool
        If True, only return metadata filepaths that have corresponding data in the local DISDRODB raw archive.
        The default is True

    Returns
    -------
    metadata_filepaths: list
        List of metadata YAML file paths

    """
    # Get all config files from the metadata directories
    list_of_base_path = []
    if data_sources:
        if isinstance(data_sources, str):
            data_sources = [data_sources]
    else:
        data_sources = ["**"]
    if campaign_names:
        if isinstance(campaign_names, str):
            campaign_names = [campaign_names]
    else:
        campaign_names = ["**"]

    for data_source in data_sources:
        for campaign_name in campaign_names:
            base_path = os.path.join(base_dir, "Raw", data_source, campaign_name)
            list_of_base_path.append(base_path)

    metadata_filepaths = []
    for base_path in list_of_base_path:
        if station_names:
            if isinstance(station_names, str):
                station_names = [station_names]
            for station_name in station_names:
                metadata_path = os.path.join(base_path, "**", "metadata", f"{station_name}.yml")
                metadata_filepaths += glob.glob(metadata_path, recursive=True)
        else:
            metadata_path = os.path.join(base_path, "**", "metadata", "*.yml")
            metadata_filepaths += glob.glob(metadata_path, recursive=True)

    return list(set(metadata_filepaths))


def _get_list_metadata_with_data(base_dir, data_sources=None, campaign_names=None, station_names=None):
    """
    Get the list of metadata filepaths that have corresponding data in the DISDRODB raw archive.

    Parameters
    ----------
    base_dir : str
        Base directory of DISDRODB
        Format: <...>/DISDRODB
    data_sources : str or list of str
        Name of data source(s) of interest.
        The name(s) must be UPPER CASE.
        The default is None
    campaign_names : str or list of str
        Name of the campaign(s) of interest.
        The name(s) must be UPPER CASE.
        The default is None
    station_names : str or list of str
        Station names of interest.
        The default is None

    Returns
    -------
    metadata_filepaths: list
        List of metadata YAML file paths

    """
    from disdrodb.api.io import available_stations

    # Check inputs
    if isinstance(data_sources, str):
        data_sources = [data_sources]
    if isinstance(campaign_names, str):
        campaign_names = [campaign_names]

    # Retrieve information of requested stations
    # --> (data_source, campaign_name, station_name)

    list_info = available_stations(
        base_dir=base_dir,
        product="RAW",
        data_sources=data_sources,
        campaign_names=campaign_names,
    )

    # If no stations available, raise an error
    if len(list_info) == 0:
        raise ValueError("No stations are available !")

    # Get metadata filepaths
    metadata_filepaths = [
        define_metadata_filepath(
            product="Raw",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            base_dir=base_dir,
            check_exists=False,
        )
        for data_source, campaign_name, station_name in list_info
    ]

    return metadata_filepaths


####--------------------------------------------------------------------------.
#### Default (empty) metadata
def _create_default_metadata_dict() -> dict:
    """Get DISDRODB metadata default values.

    Returns
    -------
    dict
        Dictionary of attributes standard
    """
    # Get valid metadata keys
    list_attrs = get_valid_metadata_keys()
    attrs = {key: "" for key in list_attrs}

    # Add default values for certain keys
    attrs["latitude"] = -9999
    attrs["longitude"] = -9999
    attrs["altitude"] = -9999
    attrs["raw_data_format"] = "txt"  # ['txt', 'netcdf']
    attrs["platform_type"] = "fixed"  # ['fixed', 'mobile']
    return attrs


def write_default_metadata(filepath: str) -> None:
    """Create default YAML metadata file at the specified filepath.

    Parameters
    ----------
    filepath : str
        File path
    """
    # Get default metadata dict
    metadata = _create_default_metadata_dict()

    # Try infer the data_source, campaign_name and station_name from filepath
    try:
        campaign_name = infer_campaign_name_from_path(filepath)
        data_source = infer_data_source_from_path(filepath)
        station_name = os.path.basename(filepath).split(".yml")[0]
        metadata["data_source"] = data_source
        metadata["campaign_name"] = campaign_name
        metadata["station_name"] = station_name
    except Exception:
        pass

    # Write the metadata
    metadata = sort_metadata_dictionary(metadata)
    write_yaml(metadata, filepath=filepath, sort_keys=False)
    return None


def create_station_metadata(data_source, campaign_name, station_name, base_dir=None, product="RAW"):
    """Create a default (semi-empty) YAML metadata file for a DISDRODB station.

    An error is raised if the file already exists !

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
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    product : str, optional
        The DISDRODB product in which to search for the metadata file.
        The default is "RAW".

    Returns
    -------
    metadata: dictionary
        The station metadata dictionary

    """
    # Define metadata filepath
    metadata_filepath = define_metadata_filepath(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        base_dir=base_dir,
        product=product,
        check_exists=False,
    )
    if os.path.exists(metadata_filepath):
        raise ValueError("A metadata YAML file already exists at {metadata_filepath}.")
    # Create metadata dir if not existing
    metadata_dir = os.path.dirname(metadata_filepath)
    os.makedirs(metadata_dir, exist_ok=True)
    # Write metadata file
    write_default_metadata(filepath=metadata_filepath)
    print(f"An empty metadata for station {station_name} has been created .")
    return None
