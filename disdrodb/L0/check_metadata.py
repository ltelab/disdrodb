#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:42:56 2022

@author: ghiggi
"""
import os
import glob
import yaml
from typing import Union

# TODO: to improve a lot !!!
# - Check metadata !!!
# - Define rules !!!


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


def identify_missing_metadata(metadata_fpaths: list, keys: Union[str, list]) -> None:
    """Identify missing metadata.

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
                print(f"Missing {key} at: ", fpath)
    return None


def identify_missing_coords(metadata_fpaths: str) -> None:
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
        longitude = metadata.get("longitude", -9999)
        latitude = metadata.get("latitude", -9999)
        # Check type validity
        if isinstance(longitude, str):
            raise TypeError(f"longitude is not defined as numeric at {fpath}.")
        if isinstance(latitude, str):
            raise TypeError(f"latitude is not defined as numeric at {fpath}.")
        # Check is not none
        if isinstance(longitude, type(None)) or isinstance(latitude, type(None)):
            print(f"Unspecified lat lon coordinates at: {fpath}")
            continue
        # Check value validity
        if longitude == -9999 or latitude == -9999:
            print(f"Missing lat lon coordinates at: {fpath}")
        elif longitude > 180 or longitude < -180:
            print("Unvalid longitude at : ", fpath)
        elif latitude > 90 or latitude < -90:
            print("Unvalid latitude at : ", fpath)
        else:
            pass
    return None


# # EXAMPLE
# ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Raw" # ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Processed"
# metadata_fpaths = glob.glob(os.path.join(ARCHIVE_DIR, "*/*/metadata/*.yml"))
# identify_missing_coords(metadata_fpaths)

# identify_missing_metadata(metadata_fpaths, keys="station_id")
# identify_missing_metadata(metadata_fpaths, keys="station_name")
# identify_missing_metadata(metadata_fpaths, keys=["station_id","station_name"])
# identify_missing_metadata(metadata_fpaths, keys="campaign_name")

# -----------------------------------------------------------------------------.
# TODO
# - station_id
# - station_number --> to be replaced by station_id
# - station_name --> if missing, use station_id

# - TODO: define dtype of metadata !!!
# - raise error if missing station_id
# - raise error if missing campaign_name
#
