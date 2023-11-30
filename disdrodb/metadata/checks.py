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
"""Check metadata."""

import os
from typing import Union

import numpy as np

from disdrodb.api.info import (
    infer_campaign_name_from_path,
    infer_data_source_from_path,
)
from disdrodb.configs import get_base_dir
from disdrodb.metadata.reader import read_station_metadata
from disdrodb.metadata.search import get_list_metadata
from disdrodb.metadata.standards import get_valid_metadata_keys
from disdrodb.utils.yaml import read_yaml

#### --------------------------------------------------------------------------.
#### Check Station Metadata


def get_metadata_missing_keys(metadata):
    """Return the DISDRODB metadata keys which are missing."""
    keys = list(metadata.keys())
    valid_keys = get_valid_metadata_keys()
    # Identify missing keys
    idx_missing_keys = np.where(np.isin(valid_keys, keys, invert=True))[0]
    missing_keys = np.array(valid_keys)[idx_missing_keys].tolist()
    return missing_keys


def get_metadata_invalid_keys(metadata):
    """Return the DISDRODB metadata keys which are not valid."""
    keys = list(metadata.keys())
    valid_keys = get_valid_metadata_keys()
    # Identify invalid keys
    idx_invalid_keys = np.where(np.isin(keys, valid_keys, invert=True))[0]
    invalid_keys = np.array(keys)[idx_invalid_keys].tolist()
    return invalid_keys


def _check_metadata_keys(metadata):
    """Check validity of metadata keys."""
    # Check all keys are valid
    invalid_keys = get_metadata_invalid_keys(metadata)
    if len(invalid_keys) > 0:
        raise ValueError(f"Invalid metadata keys: {invalid_keys}")
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
    from disdrodb.api.checks import check_sensor_name

    sensor_name = metadata["sensor_name"]
    check_sensor_name(sensor_name=sensor_name)
    return None


def check_metadata_compliance(data_source, campaign_name, station_name, base_dir=None, product="RAW"):
    """Check DISDRODB metadata compliance."""
    from disdrodb.l0.l0_reader import _check_metadata_reader

    metadata = read_station_metadata(
        base_dir=base_dir,
        product=product,
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


#### --------------------------------------------------------------------------.
#### Metadata Archive Missing Information


def _check_lonlat_type(longitude, latitude):
    # Check type validity
    if isinstance(longitude, str):
        raise TypeError("longitude is not defined as numeric.")
    if isinstance(latitude, str):
        raise TypeError("latitude is not defined as numeric.")
    # Check is not none
    if isinstance(longitude, type(None)) or isinstance(latitude, type(None)):
        raise ValueError("Unspecified longitude and latitude coordinates.")


def _check_lonlat_validity(longitude, latitude):
    if longitude == -9999 or latitude == -9999:
        raise ValueError("Missing lat lon coordinates (-9999).")
    elif longitude > 180 or longitude < -180:
        raise ValueError("Invalid longitude (outside [-180, 180])")
    elif latitude > 90 or latitude < -90:
        raise ValueError("Invalid latitude (outside [-90, 90])")
    else:
        pass


def check_metadata_geolocation(metadata) -> None:
    """Identify metadata with missing or wrong geolocation."""
    # Get longitude, latitude and platform type
    longitude = metadata.get("longitude")
    latitude = metadata.get("latitude")
    platform_type = metadata.get("platform_type")
    # Check type validity
    _check_lonlat_type(longitude=longitude, latitude=latitude)
    # Check value validity
    # - If mobile platform
    if platform_type == "mobile":
        if longitude != -9999 or latitude != -9999:
            raise ValueError("For mobile platform_type, specify latitude and longitude -9999")
    # - If fixed platform
    else:
        _check_lonlat_validity(longitude=longitude, latitude=latitude)
    return None


def identify_missing_metadata_coords(metadata_filepaths: str) -> None:
    """Identify missing coordinates.

    Parameters
    ----------
    metadata_filepaths : str
        Input YAML file path.

    Raises
    ------
    TypeError
        Error if latitude or longitude coordinates are not present or are wrongly formatted.

    """
    for filepath in metadata_filepaths:
        metadata = read_yaml(filepath)
        check_metadata_geolocation(metadata)
    return None


def identify_empty_metadata_keys(metadata_filepaths: list, keys: Union[str, list]) -> None:
    """Identify empty metadata keys.

    Parameters
    ----------
    metadata_filepaths : str
        Input YAML file path.
    keys : Union[str,list]
        Attributes to verify the presence.
    """

    if isinstance(keys, str):
        keys = [keys]

    for filepath in metadata_filepaths:
        for key in keys:
            metadata = read_yaml(filepath)
            if len(str(metadata.get(key, ""))) == 0:  # ensure is string to avoid error
                print(f"Empty {key} at: ", filepath)
    return None


#### --------------------------------------------------------------------------.
#### Check Metadata Archive


def check_archive_metadata_keys(base_dir: str = None) -> bool:
    """Check that all metadata files have valid keys

    Parameters
    ----------
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")

        metadata = read_station_metadata(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        try:
            _check_metadata_keys(metadata)
        except Exception as e:
            print(f"Error for {data_source} {campaign_name} {station_name}.")
            print(f"The error is: {e}.")
            is_valid = False

    return is_valid


def check_archive_metadata_campaign_name(base_dir: str = None) -> bool:
    """Check metadata campaign_name.

    Parameters
    ----------
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")

        metadata = read_station_metadata(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        try:
            _check_metadata_campaign_name(metadata, expected_name=campaign_name)
        except Exception as e:
            is_valid = False
            print(f"Error for {data_source} {campaign_name} {station_name}.")
            print(f"The error is: {e}.")
    return is_valid


def check_archive_metadata_data_source(base_dir: str = None) -> bool:
    """Check metadata data_source.

    Parameters
    ----------
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")

        metadata = read_station_metadata(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        try:
            _check_metadata_data_source(metadata, expected_name=data_source)
        except Exception as e:
            is_valid = False
            print(f"Error for {data_source} {campaign_name} {station_name}.")
            print(f"The error is: {e}.")
    return is_valid


def check_archive_metadata_sensor_name(base_dir: str = None) -> bool:
    """Check metadata sensor name.

    Parameters
    ----------
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")

        metadata = read_station_metadata(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        try:
            _check_metadata_sensor_name(metadata)
        except Exception as e:
            is_valid = False
            print(f"Error for {data_source} {campaign_name} {station_name}.")
            print(f"The error is: {e}.")
    return is_valid


def check_archive_metadata_station_name(base_dir: str = None) -> bool:
    """Check metadata station name.

    Parameters
    ----------
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")

        metadata = read_station_metadata(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        try:
            _check_metadata_station_name(metadata, expected_name=station_name)
        except Exception as e:
            is_valid = False
            print(f"Error for {data_source} {campaign_name} {station_name}.")
            print(f"The error is: {e}.")
    return is_valid


def check_archive_metadata_reader(base_dir: str = None) -> bool:
    """Check if the reader key is available and there is the associated reader.

    Parameters
    ----------
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    from disdrodb.l0.l0_reader import _check_metadata_reader

    is_valid = True
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")

        metadata = read_station_metadata(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        try:
            _check_metadata_reader(metadata)
        except Exception as e:
            is_valid = False
            print(f"Error for {data_source} {campaign_name} {station_name}.")
            print(f"The error is: {e}.")
    return is_valid


def check_archive_metadata_compliance(base_dir: str = None, raise_error=False):
    """Check the archive metadata compliance.

    Parameters
    ----------
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.
    raise_error: bool (optional)
        Whether to raise an error and interrupt the archive check if a
        metadata is not compliant. The default is False.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")
        # Check compliance
        try:
            check_metadata_compliance(
                base_dir=base_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
            )
        except Exception as e:
            is_valid = False
            msg = f"Error for {data_source} {campaign_name} {station_name}."
            msg = msg + f"The error is: {e}."
            if raise_error:
                raise ValueError(msg)
            else:
                print(msg)

    return is_valid


def check_archive_metadata_geolocation(base_dir: str = None):
    """Check the metadata files have missing or wrong geolocation..

    Parameters
    ----------
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")

        metadata = read_station_metadata(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        try:
            check_metadata_geolocation(metadata)
        except Exception as e:
            is_valid = False
            print(f"Missing information for {data_source} {campaign_name} {station_name}.")
            print(f"The error is: {e}.")
    return is_valid
