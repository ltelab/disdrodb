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
"""Routines to read, check and write DISDRODB metadata."""

import os

from disdrodb.api.info import infer_campaign_name_from_path, infer_data_source_from_path
from disdrodb.configs import get_base_dir
from disdrodb.metadata.manipulation import sort_metadata_dictionary
from disdrodb.metadata.standards import get_valid_metadata_keys
from disdrodb.utils.yaml import read_yaml, write_yaml

####--------------------------------------------------------------------------.
#### Metadata reader & writers


def read_metadata(campaign_dir: str, station_name: str) -> dict:
    """Read YAML metadata file.

    Parameters
    ----------
    raw_dir : str
        Path of the raw directory
    station_name : int
        Id of the station.

    Returns
    -------
    dict
        Dictionary  of the metadata.
    """

    metadata_fpath = os.path.join(campaign_dir, "metadata", station_name + ".yml")
    metadata = read_yaml(metadata_fpath)
    return metadata


####--------------------------------------------------------------------------.
#### Default (empty) metadata
def _get_default_metadata_dict() -> dict:
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


def _write_metadata(metadata, fpath):
    """Write dictionary to YAML file."""
    metadata = sort_metadata_dictionary(metadata)
    write_yaml(
        dictionary=metadata,
        fpath=fpath,
        sort_keys=False,
    )
    return None


def write_default_metadata(fpath: str) -> None:
    """Create default YAML metadata file at the specified filepath.

    Parameters
    ----------
    fpath : str
        File path
    """
    # Get default metadata dict
    metadata = _get_default_metadata_dict()
    # Try infer the data_source, campaign_name and station_name from fpath
    try:
        campaign_name = infer_campaign_name_from_path(fpath)
        data_source = infer_data_source_from_path(fpath)
        station_name = os.path.basename(fpath).split(".yml")[0]
        metadata["data_source"] = data_source
        metadata["campaign_name"] = campaign_name
        metadata["station_name"] = station_name
    except Exception:
        pass
    # Write the metadata
    _write_metadata(metadata=metadata, fpath=fpath)
    return None


def create_campaign_default_metadata(
    campaign_name,
    data_source,
    base_dir=None,
):
    """Create default YAML metadata files for all stations within a campaign.

    Use the function with caution to avoid overwrite existing YAML files.
    """
    base_dir = get_base_dir(base_dir)
    data_dir = os.path.join(base_dir, "Raw", data_source, campaign_name, "data")
    metadata_dir = os.path.join(base_dir, "Raw", data_source, campaign_name, "metadata")
    station_names = os.listdir(data_dir)
    for station_name in station_names:
        metadata_fpath = os.path.join(metadata_dir, station_name + ".yml")
        write_default_metadata(fpath=metadata_fpath)
    print(f"The default metadata were created for stations {station_names}.")
    return None
