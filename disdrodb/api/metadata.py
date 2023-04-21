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
import yaml
import glob
from disdrodb.api.io import _get_disdrodb_directory


def _read_yaml_file(fpath):
    """Read a YAML file into dictionary."""
    with open(fpath, "r") as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def _write_yaml_file(dictionary, fpath, sort_keys=False):
    """Write dictionary to YAML file."""
    with open(fpath, "w") as f:
        yaml.dump(dictionary, f, sort_keys=sort_keys)
    return None


def read_station_metadata(disdrodb_dir, product_level, data_source, campaign_name, station_name):
    """Open the station metadata YAML file into a dictionary."""
    # Retrieve campaign directory
    campaign_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exist=True,
    )
    # Define metadata filepath
    fpath = os.path.join(campaign_dir, "metadata", f"{station_name}.yml")

    # Check the file exists
    if not os.path.exists(fpath):
        raise ValueError(f"The metadata file for {station_name} at {fpath} does not exists.")

    # Read the metadata file
    with open(fpath, "r") as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def get_list_metadata(
    disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=True
):
    """
    Get the list of metadata filepaths in the DISDRODB raw archive.

    Parameters
    ----------
    disdrodb_dir : str
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
    metadata_fpaths: list
        List of metadata YAML file paths

    """
    if with_stations_data:
        return _get_list_metadata_with_data(disdrodb_dir, data_sources, campaign_names, station_names)
    else:
        return _get_list_all_metadata(disdrodb_dir, data_sources, campaign_names, station_names)


def _get_list_all_metadata(disdrodb_dir, data_sources=None, campaign_names=None, station_names=None):
    """
    Get the list of metadata filepaths in the DISDRODB raw archive.

    Parameters
    ----------
    disdrodb_dir : str
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
    metadata_fpaths: list
        List of metadata YAML file paths

    """
    # Get all config files from the metadata folders
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
            base_path = os.path.join(disdrodb_dir, "Raw", data_source, campaign_name)
            list_of_base_path.append(base_path)

    metadata_folder_name = "metadata"

    metadata_fpaths = []

    for base_path in list_of_base_path:
        if station_names:
            if isinstance(station_names, str):
                station_names = [station_names]
            for station_name in station_names:
                metadata_path = os.path.join(base_path, "**", metadata_folder_name, f"{station_name}.yml")
                metadata_fpaths += glob.glob(metadata_path, recursive=True)
        else:
            metadata_path = os.path.join(base_path, "**", metadata_folder_name, "*.yml")
            metadata_fpaths += glob.glob(metadata_path, recursive=True)

    return list(set(metadata_fpaths))


def _get_list_metadata_with_data(disdrodb_dir, data_sources=None, campaign_names=None, station_names=None):
    """
    Get the list of metadata filepaths that have corresponding data in the DISDRODB raw archive.

    Parameters
    ----------
    disdrodb_dir : str
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
    metadata_fpaths: list
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
        disdrodb_dir,
        product_level="RAW",
        data_sources=data_sources,
        campaign_names=campaign_names,
    )

    # If no stations available, raise an error
    if len(list_info) == 0:
        raise ValueError("No stations are available !")

    # Get metadata filepaths
    metadata_fpaths = [
        os.path.join(disdrodb_dir, "Raw", data_source, campaign_name, "metadata", (station_name + ".yml"))
        for data_source, campaign_name, station_name in list_info
    ]

    return metadata_fpaths
