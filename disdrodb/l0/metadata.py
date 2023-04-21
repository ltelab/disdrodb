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
import numpy as np
from disdrodb.api.metadata import _read_yaml_file, _write_yaml_file
from disdrodb.l0.io import get_campaign_name, get_data_source


####--------------------------------------------------------------------------.
#### Define valid metadata keys
def get_valid_metadata_keys() -> list:
    """Get DISDRODB valid metadata list.

    Returns
    -------
    list
        List of valid metadata keys
    """
    list_attrs = [
        ## Mandatory fields
        "data_source",
        "campaign_name",
        "station_name",
        "sensor_name",
        "reader",
        "raw_data_format",  # 'txt', 'netcdf'
        "platform_type",  # 'fixed', 'mobile'
        ## Source
        "source",
        "source_convention",
        "source_processing_date",
        ## Description
        "title",
        "description",
        "project_name",
        "keywords",
        "summary",
        "history",
        "comment",
        "station_id",
        "location",
        "country",
        "continent",
        ## Deployment Info
        "latitude",  # in degrees North
        "longitude",  # in degrees East
        "altitude",  # in meter above sea level
        "deployment_status",  # 'ended', 'ongoing'
        "deployment mode",  # 'land', 'ship', 'truck', 'cable'
        "platform_protection",  # 'shielded', 'unshielded'
        "platform_orientation",  # [0-360] from N (clockwise)
        ## Sensor info
        "sensor_long_name",
        "sensor_manufacturer",
        "sensor_wavelength",
        "sensor_serial_number",
        "firmware_iop",
        "firmware_dsp",
        "firmware_version",
        "sensor_beam_length",
        "sensor_beam_width",
        "sensor_nominal_width",  # ?
        ## effective_measurement_area ?  # 0.54 m^2
        "measurement_interval",  # sampling_interval ? [in seconds]
        "calibration_sensitivity",
        "calibration_certification_date",
        "calibration_certification_url",
        ## Attribution
        "contributors",
        "authors",
        "authors_url",
        "contact",
        "contact_information",
        "acknowledgement",  # acknowledgements?
        "references",
        "documentation",
        "website",
        "institution",
        "source_repository",
        "license",
        "doi",
    ]
    return list_attrs


####--------------------------------------------------------------------------.
#### Metadata reader & writers


def sort_metadata_dictionary(metadata):
    """Sort the keys of the metadata dictionary by valid_metadata_keys list order."""
    list_metadata_keys = get_valid_metadata_keys()
    metadata = {k: metadata[k] for k in list_metadata_keys}
    return metadata


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
        Dictionnary of the metadata.
    """

    metadata_fpath = os.path.join(campaign_dir, "metadata", station_name + ".yml")
    metadata = _read_yaml_file(metadata_fpath)
    return metadata


def write_metadata(metadata, fpath):
    """Write dictionary to YAML file."""
    metadata = sort_metadata_dictionary(metadata)
    _write_yaml_file(
        dictionary=metadata,
        fpath=fpath,
        sort_keys=False,
    )
    return None


####--------------------------------------------------------------------------.
#### Default (empty) metadata
def get_default_metadata_dict() -> dict:
    """Get DISDRODB metadata default values.

    Returns
    -------
    dict
        Dictionary of attibutes standard
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


def write_default_metadata(fpath: str) -> None:
    """Create default YAML metadata file at the specified filepath.

    Parameters
    ----------
    fpath : str
        File path
    """
    # Get default metadata dict
    metadata = get_default_metadata_dict()
    # Try infer the data_source, campaign_name and station_name from fpath
    try:
        campaign_name = get_campaign_name(fpath)
        data_source = get_data_source(fpath)
        station_name = os.path.basename(fpath).split(".yml")[0]
        metadata["data_source"] = data_source
        metadata["campaign_name"] = campaign_name
        metadata["station_name"] = station_name
    except Exception:
        pass
    # Write the metadata
    write_metadata(metadata=metadata, fpath=fpath)
    return None


def create_campaign_default_metadata(
    disdrodb_dir,
    campaign_name,
    data_source,
):
    """Create default YAML metadata files for all stations within a campaign.

    Use the function with caution to avoid overwrite existing YAML files.
    """
    data_dir = os.path.join(disdrodb_dir, "Raw", data_source, campaign_name, "data")
    metadata_dir = os.path.join(disdrodb_dir, "Raw", data_source, campaign_name, "metadata")
    station_names = os.listdir(data_dir)
    for station_name in station_names:
        metadata_fpath = os.path.join(metadata_dir, station_name + ".yml")
        write_default_metadata(fpath=metadata_fpath)
    print(f"The default metadata were created for stations {station_names}.")
    return None


####--------------------------------------------------------------------------.
#### Check metadata file


def get_metadata_missing_keys(metadata):
    """Return the DISDRODB metadata keys which are missing."""
    keys = list(metadata.keys())
    valid_keys = get_valid_metadata_keys()
    # Identify missing keys
    idx_missing_keys = np.where(np.isin(valid_keys, keys, invert=True))[0]
    missing_keys = np.array(valid_keys)[idx_missing_keys].tolist()
    return missing_keys


def get_metadata_unvalid_keys(metadata):
    """Return the DISDRODB metadata keys which are not valid."""
    keys = list(metadata.keys())
    valid_keys = get_valid_metadata_keys()
    # Identify unvalid keys
    idx_unvalid_keys = np.where(np.isin(keys, valid_keys, invert=True))[0]
    unvalid_keys = np.array(keys)[idx_unvalid_keys].tolist()
    return unvalid_keys


def _check_metadata_keys(metadata):
    """Check validity of metadata keys."""
    # Check all keys are valid
    unvalid_keys = get_metadata_unvalid_keys(metadata)
    if len(unvalid_keys) > 0:
        raise ValueError(f"Unvalid metadata keys: {unvalid_keys}")
    # Check no keys are missing
    missing_keys = get_metadata_missing_keys(metadata)
    if len(missing_keys) > 0:
        raise ValueError(f"Missing metadata keys: {missing_keys}")
    return None


def _check_metadata_values(metadata):
    """Check validity of metadata values

    If null is specified in the YAML files (or None in the dict) raise error.
    """
    for key, value in metadata.items():
        if isinstance(value, type(None)):
            raise ValueError(f"The metadata key {key} has None or null value. Use '' instead.")
    return None


def _check_metadata_campaign_name(metadata, expected_name):
    """Check metadata campaign_name."""
    if "campaign_name" not in metadata:
        raise ValueError("The metadata file does not contain the 'campaign_name' key.")
    campaign_name = metadata["campaign_name"]
    if campaign_name == "":
        raise ValueError("The 'campaign_name' key in the metadata is empty.")
    if campaign_name != expected_name:
        raise ValueError(
            f"The campaign_name in the metadata is '{campaign_name}' but the campaign directory is '{expected_name}'"
        )
    return None


def _check_metadata_data_source(metadata, expected_name):
    """Check metadata data_source."""
    if "data_source" not in metadata:
        raise ValueError("The metadata file does not contain the 'data_source' key.")
    data_source = metadata["data_source"]
    if data_source == "":
        raise ValueError("The 'data_source' key in the metadata is empty.")
    if data_source != expected_name:
        raise ValueError(
            f"The data_source in the metadata is '{data_source}' but the data_source directory is '{expected_name}'"
        )
    return None


def _check_metadata_station_name(metadata, expected_name):
    """Check metadata station name.

    This function does not check that data are available for the station!"""
    if "station_name" not in metadata:
        raise ValueError("The metadata file does not contain the 'station_name' key.")
    station_name = metadata["station_name"]
    if not isinstance(station_name, str):
        raise ValueError("The 'station_name' key in the metadata is not defined as a string!")
    if station_name == "":
        raise ValueError("The 'station_name' key in the metadata is empty.")
    if station_name != expected_name:
        raise ValueError(
            f"The station_name in the metadata is '{station_name}' but the metadata file is named '{expected_name}.yml'"
        )
    return None


def _check_metadata_sensor_name(metadata):
    from disdrodb.l0.check_standards import check_sensor_name

    sensor_name = metadata["sensor_name"]
    check_sensor_name(sensor_name=sensor_name)
    return None


def check_metadata_compliance(disdrodb_dir, data_source, campaign_name, station_name):
    """Check DISDRODB metadata compliance."""
    from disdrodb.l0.l0_reader import _check_metadata_reader
    from disdrodb.api.metadata import read_station_metadata

    metadata = read_station_metadata(
        disdrodb_dir=disdrodb_dir,
        product_level="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    _check_metadata_keys(metadata)
    _check_metadata_values(metadata)
    _check_metadata_campaign_name(metadata, expected_name=campaign_name)
    _check_metadata_data_source(metadata, expected_name=data_source)
    _check_metadata_station_name(metadata, expected_name=station_name)
    _check_metadata_sensor_name(metadata)
    _check_metadata_reader(metadata)
    return None


####--------------------------------------------------------------------------.
#### Metadata manipulation tools
def remove_unvalid_metadata_keys(metadata):
    """Remove unvalid keys from the metadata dictionary."""
    unvalid_keys = get_metadata_unvalid_keys(metadata)
    for k in unvalid_keys:
        _ = metadata.pop(k)
    return metadata


def add_missing_metadata_keys(metadata):
    """Add missing keys to the metadata dictionary."""
    missing_keys = get_metadata_missing_keys(metadata)
    for k in missing_keys:
        metadata[k] = ""
    return metadata
