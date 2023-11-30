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
"""Routines to write the DISDRODB Metadata."""

import os

from disdrodb.api.path import define_metadata_filepath
from disdrodb.metadata.manipulation import sort_metadata_dictionary
from disdrodb.metadata.standards import get_valid_metadata_keys
from disdrodb.utils.yaml import write_yaml


def get_default_metadata_dict() -> dict:
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


def create_station_metadata(data_source, campaign_name, station_name, base_dir=None, product="RAW"):
    """Write a default (semi-empty) YAML metadata file for a DISDRODB station.

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

    # Get default metadata dict
    metadata = get_default_metadata_dict()

    # Try infer the data_source, campaign_name and station_name from filepath
    metadata["data_source"] = data_source
    metadata["campaign_name"] = campaign_name
    metadata["station_name"] = station_name

    # Write the metadata
    metadata = sort_metadata_dictionary(metadata)
    write_yaml(metadata, filepath=metadata_filepath, sort_keys=False)

    print(f"An empty default metadata YAML file for station {station_name} has been created .")
    return None
