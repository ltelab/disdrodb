#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from typing import Union
from disdrodb.l0.io import (
    get_disdrodb_dir,
    get_data_source,
    get_campaign_name,
)
from disdrodb.api.metadata import read_station_metadata
from disdrodb.l0.l0_reader import _check_metadata_reader
from disdrodb.l0.metadata import (
    _check_metadata_keys,
    _check_metadata_data_source,
    _check_metadata_campaign_name,
    _check_metadata_station_name,
    _check_metadata_sensor_name,
    check_metadata_compliance,
)

from disdrodb.api.metadata import get_list_metadata

#### --------------------------------------------------------------------------.


def read_yaml(fpath: str) -> dict:
    """Read YAML file.

    Parameters
    ----------
    fpath : str
        Input YAML file path.

    Returns
    -------
    dict
        Attributes read from the YAML file.
    """
    with open(fpath, "r") as f:
        attrs = yaml.safe_load(f)
    return attrs


#### --------------------------------------------------------------------------.
#### Metadata Archive Missing Information


def check_metadata_geolocation(metadata) -> None:
    """Identify metadata with missing or wrong geolocation."""
    # Get longitude, latitude and platform type
    longitude = metadata.get("longitude")
    latitude = metadata.get("latitude")
    platform_type = metadata.get("platform_type")
    # Check type validity
    if isinstance(longitude, str):
        raise TypeError("longitude is not defined as numeric.")
    if isinstance(latitude, str):
        raise TypeError("latitude is not defined as numeric.")
    # Check is not none
    if isinstance(longitude, type(None)) or isinstance(latitude, type(None)):
        raise ValueError("Unspecified longitude and latitude coordinates.")
    else:
        # Check value validity
        # - If mobile platform
        if platform_type == "mobile":
            if longitude != -9999 or latitude != -9999:
                raise ValueError("For mobile platform_type, specify latitude and longitude -9999")
        # - If fixed platform
        else:
            if longitude == -9999 or latitude == -9999:
                raise ValueError("Missing lat lon coordinates (-9999).")
            elif longitude > 180 or longitude < -180:
                raise ValueError("Unvalid longitude (outside [-180, 180])")
            elif latitude > 90 or latitude < -90:
                raise ValueError("Unvalid latitude (outside [-90, 90])")
            else:
                pass
    return None


def identify_missing_metadata_coords(metadata_fpaths: str) -> None:
    """Identify missing coordinates.

    Parameters
    ----------
    metadata_fpaths : str
        Input YAML file path.

    Raises
    ------
    TypeError
        Error if latitude or longitude coordinates are not present or are wrongly formatted.

    """
    for fpath in metadata_fpaths:
        metadata = read_yaml(fpath)
        check_metadata_geolocation(metadata)
    return None


def identify_empty_metadata_keys(metadata_fpaths: list, keys: Union[str, list]) -> None:
    """Identify empty metadata keys.

    Parameters
    ----------
    metadata_fpaths : str
        Input YAML file path.
    keys : Union[str,list]
        Attributes to verify the presence.
    """

    if isinstance(keys, str):
        keys = [keys]

    for fpath in metadata_fpaths:
        for key in keys:
            metadata = read_yaml(fpath)
            if len(str(metadata.get(key, ""))) == 0:  # ensure is string to avoid error
                print(f"Empty {key} at: ", fpath)
    return None


def get_archive_metadata_key_value(disdrodb_dir: str, key: str, return_tuple: bool = True):
    """Return the values of a metadata key for all the archive.
    Parameters
    ----------
    disdrodb_dir : str
        Path to the disdrodb directory.
    key : str
        Metadata key.
    return_tuple : bool, optional
        if True, returns a tuple of values with station, campaign and data source name (default is True)
        if False, returns a list of values without station, campaign and data source name
    Returns
    -------
    list or tuple
        List or tuple of values of the metadata key.
    """

    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    list_info = []
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")
        metadata = read_station_metadata(
            disdrodb_dir=disdrodb_dir,
            product_level="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        value = metadata[key]
        info = (data_source, campaign_name, station_name, value)
        list_info.append(info)
    if not return_tuple:
        list_info = [info[3] for info in list_info]
    return list_info


#### --------------------------------------------------------------------------.
#### Metadata Archive Checks
def check_archive_metadata_keys(disdrodb_dir: str) -> bool:
    """Check that all metadata files have valid keys

    Parameters
    ----------
    disdrodb_dir : str
        Path to the disdrodb directory.
    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True

    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")

        metadata = read_station_metadata(
            disdrodb_dir=disdrodb_dir,
            product_level="RAW",
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


def check_archive_metadata_campaign_name(disdrodb_dir) -> bool:
    """Check metadata campaign_name.

    Parameters
    ----------
    disdrodb_dir : str
        Path to the disdrodb directory.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")

        metadata = read_station_metadata(
            disdrodb_dir=disdrodb_dir,
            product_level="RAW",
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


def check_archive_metadata_data_source(disdrodb_dir) -> bool:
    """Check metadata data_source.

    Parameters
    ----------
    disdrodb_dir : str
        Path to the disdrodb directory.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")

        metadata = read_station_metadata(
            disdrodb_dir=disdrodb_dir,
            product_level="RAW",
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


def check_archive_metadata_sensor_name(disdrodb_dir) -> bool:
    """Check metadata sensor name.

    Parameters
    ----------
    disdrodb_dir : str
        Path to the disdrodb directory.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")

        metadata = read_station_metadata(
            disdrodb_dir=disdrodb_dir,
            product_level="RAW",
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


def check_archive_metadata_station_name(disdrodb_dir) -> bool:
    """Check metadata station name.

    Parameters
    ----------
    disdrodb_dir : str
        Path to the disdrodb directory.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")

        metadata = read_station_metadata(
            disdrodb_dir=disdrodb_dir,
            product_level="RAW",
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


def check_archive_metadata_reader(disdrodb_dir: str) -> bool:
    """Check if the reader key is available and there is the associated reader.

    Parameters
    ----------
    disdrodb_dir : str
        Path to the disdrodb directory.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """

    is_valid = True
    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")

        metadata = read_station_metadata(
            disdrodb_dir=disdrodb_dir,
            product_level="RAW",
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


def check_archive_metadata_compliance(disdrodb_dir):
    is_valid = True
    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")
        try:
            check_metadata_compliance(
                disdrodb_dir=disdrodb_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
            )
        except Exception as e:
            is_valid = False
            print(f"Error for {data_source} {campaign_name} {station_name}.")
            print(f"The error is: {e}.")
    return is_valid


def check_archive_metadata_geolocation(disdrodb_dir):
    """Check the metadata files have missing or wrong geolocation..

    Parameters
    ----------
    disdrodb_dir : str
        Path to the disdrodb directory.

    Returns
    -------
    bool
        If the check succeeds, the result is True, and if it fails, the result is False.
    """
    is_valid = True
    list_metadata_paths = get_list_metadata(
        disdrodb_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    for fpath in list_metadata_paths:
        disdrodb_dir = get_disdrodb_dir(fpath)
        data_source = get_data_source(fpath)
        campaign_name = get_campaign_name(fpath)
        station_name = os.path.basename(fpath).replace(".yml", "")

        metadata = read_station_metadata(
            disdrodb_dir=disdrodb_dir,
            product_level="RAW",
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
