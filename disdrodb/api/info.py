#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import numpy as np
from trollsift import Parser

####---------------------------------------------------------------------------
########################
#### FNAME PATTERNS ####
########################
DISDRODB_FNAME_PATTERN = (
    "{product_level:s}.{campaign_name:s}.{station_name:s}.s{start_time:%Y%m%d%H%M%S}.e{end_time:%Y%m%d%H%M%S}"
    ".{version:s}.{data_format:s}"
)

####---------------------------------------------------------------------------.
##########################
#### Filename parsers ####
##########################
fname = "L0A.LOCARNO_2018.60.s20180625004331.e20180711010000.dev.parquet"


def _parse_fname(fname):
    """Parse the filename with trollsift."""
    # Retrieve information from filename
    p = Parser(DISDRODB_FNAME_PATTERN)
    info_dict = p.parse(fname)
    return info_dict


def _get_info_from_filename(fname):
    """Retrieve file information dictionary from filename."""
    # Try to parse the filename
    try:
        info_dict = _parse_fname(fname)
    except ValueError:
        raise ValueError(f"{fname} can not be parsed. Report the issue.")
    # Return info dictionary
    return info_dict


def get_info_from_filepath(fpath):
    """Retrieve file information dictionary from filepath."""
    if not isinstance(fpath, str):
        raise TypeError("'fpath' must be a string.")
    fname = os.path.basename(fpath)
    return _get_info_from_filename(fname)


def get_key_from_filepath(fpath, key):
    """Extract specific key information from a list of filepaths."""
    value = get_info_from_filepath(fpath)[key]
    return value


def get_key_from_filepaths(fpaths, key):
    """Extract specific key information from a list of filepaths."""
    if isinstance(fpaths, str):
        fpaths = [fpaths]
    return [get_key_from_filepath(fpath, key=key) for fpath in fpaths]


####--------------------------------------------------------------------------.
####################################
#### DISDRODB File Informations ####
####################################


def _get_version_from_filepath(filepath, integer=False):
    version = get_key_from_filepath(filepath, key="version")
    if integer:
        version = int(re.findall("\d+", version)[0])
    return version


def get_version_from_filepaths(filepaths, integer=False):
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    list_version = [_get_version_from_filepath(fpath, integer=integer) for fpath in filepaths]
    return list_version


def get_campaign_name_from_filepaths(filepaths):
    list_id = get_key_from_filepaths(filepaths, key="campaign_name")
    return list_id


def get_station_name_from_filepaths(filepaths):
    list_id = get_key_from_filepaths(filepaths, key="station_name")
    return list_id


def get_product_level_from_filepaths(filepaths):
    list_id = get_key_from_filepaths(filepaths, key="product_level")
    return list_id


def get_start_time_from_filepaths(filepaths):
    list_start_time = get_key_from_filepaths(filepaths, key="start_time")
    return list_start_time


def get_end_time_from_filepaths(filepaths):
    list_end_time = get_key_from_filepaths(filepaths, key="end_time")
    return list_end_time


def get_start_end_time_from_filepaths(filepaths):
    list_start_time = get_key_from_filepaths(filepaths, key="start_time")
    list_end_time = get_key_from_filepaths(filepaths, key="end_time")
    return np.array(list_start_time), np.array(list_end_time)


####--------------------------------------------------------------------------.
