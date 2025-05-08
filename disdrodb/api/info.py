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
"""Retrieve file information from DISDRODB products file names and filepaths."""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from trollsift import Parser

from disdrodb.utils.time import acronym_to_seconds

####---------------------------------------------------------------------------
########################
#### FNAME PATTERNS ####
########################
DISDRODB_FNAME_L0_PATTERN = (
    "{product:s}.{campaign_name:s}.{station_name:s}.s{start_time:%Y%m%d%H%M%S}.e{end_time:%Y%m%d%H%M%S}"
    ".{version:s}.{data_format:s}"
)
DISDRODB_FNAME_L2E_PATTERN = (  # also L0C and L1 --> accumulation_acronym = sample_interval
    "{product:s}.{accumulation_acronym}.{campaign_name:s}.{station_name:s}.s{start_time:%Y%m%d%H%M%S}.e{end_time:%Y%m%d%H%M%S}"
    ".{version:s}.{data_format:s}"
)

DISDRODB_FNAME_L2M_PATTERN = (
    "{product:s}_{subproduct:s}.{accumulation_acronym}.{campaign_name:s}.{station_name:s}.s{start_time:%Y%m%d%H%M%S}.e{end_time:%Y%m%d%H%M%S}"
    ".{version:s}.{data_format:s}"
)

####---------------------------------------------------------------------------.
##########################
#### Filename parsers ####
##########################


def _parse_filename(filename):
    """Parse the filename with trollsift."""
    if filename.startswith("L0A") or filename.startswith("L0B"):
        p = Parser(DISDRODB_FNAME_L0_PATTERN)
        info_dict = p.parse(filename)
    elif filename.startswith("L2E") or filename.startswith("L1") or filename.startswith("L0C"):
        p = Parser(DISDRODB_FNAME_L2E_PATTERN)
        info_dict = p.parse(filename)
    elif filename.startswith("L2M"):
        p = Parser(DISDRODB_FNAME_L2M_PATTERN)
        info_dict = p.parse(filename)
    else:
        raise ValueError("Not a DISDRODB product file.")
    return info_dict


def _get_info_from_filename(filename):
    """Retrieve file information dictionary from filename."""
    # Try to parse the filename
    try:
        info_dict = _parse_filename(filename)
    except ValueError:
        raise ValueError(f"{filename} can not be parsed. Report the issue.")

    # Add additional information to info dictionary
    if "accumulation_acronym" in info_dict:
        info_dict["sample_interval"] = acronym_to_seconds(info_dict["accumulation_acronym"])

    # Return info dictionary
    return info_dict


def get_info_from_filepath(filepath):
    """Retrieve file information dictionary from filepath."""
    if not isinstance(filepath, str):
        raise TypeError("'filepath' must be a string.")
    filename = os.path.basename(filepath)
    return _get_info_from_filename(filename)


def get_key_from_filepath(filepath, key):
    """Extract specific key information from a list of filepaths."""
    value = get_info_from_filepath(filepath)[key]
    return value


def get_key_from_filepaths(filepaths, key):
    """Extract specific key information from a list of filepaths."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    return [get_key_from_filepath(filepath, key=key) for filepath in filepaths]


####--------------------------------------------------------------------------.
###################################
#### DISDRODB File Information ####
###################################


def _get_version_from_filepath(filepath):
    version = get_key_from_filepath(filepath, key="version")
    return version


def get_version_from_filepaths(filepaths):
    """Return the DISDROB product version of the specified files."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    list_version = [_get_version_from_filepath(filepath) for filepath in filepaths]
    return list_version


def get_campaign_name_from_filepaths(filepaths):
    """Return the DISDROB campaign name of the specified files."""
    list_id = get_key_from_filepaths(filepaths, key="campaign_name")
    return list_id


def get_station_name_from_filepaths(filepaths):
    """Return the DISDROB station name of the specified files."""
    list_id = get_key_from_filepaths(filepaths, key="station_name")
    return list_id


def get_product_from_filepaths(filepaths):
    """Return the DISDROB product name of the specified files."""
    list_id = get_key_from_filepaths(filepaths, key="product")
    return list_id


def get_start_time_from_filepaths(filepaths):
    """Return the start time of the specified files."""
    list_start_time = get_key_from_filepaths(filepaths, key="start_time")
    return list_start_time


def get_end_time_from_filepaths(filepaths):
    """Return the end time of the specified files."""
    list_end_time = get_key_from_filepaths(filepaths, key="end_time")
    return list_end_time


def get_start_end_time_from_filepaths(filepaths):
    """Return the start and end time of the specified files."""
    list_start_time = get_key_from_filepaths(filepaths, key="start_time")
    list_end_time = get_key_from_filepaths(filepaths, key="end_time")
    return np.array(list_start_time).astype("M8[s]"), np.array(list_end_time).astype("M8[s]")


def get_sample_interval_from_filepaths(filepaths):
    """Return the sample interval of the specified files."""
    list_accumulation_acronym = get_key_from_filepaths(filepaths, key="accumulation_acronym")
    list_sample_interval = [acronym_to_seconds(s) for s in list_accumulation_acronym]
    return list_sample_interval


####--------------------------------------------------------------------------.
###################################
#### DISDRODB Tree Components  ####
###################################


def infer_disdrodb_tree_path_components(path: str) -> list:
    """Return a list with the component of a DISDRODB path ``disdrodb_path``.

    Parameters
    ----------
    path : str
        Directory or file path within the DISDRODB archive.

    Returns
    -------
    list
        Path element of the DISDRODB archive.
        Format: [``data_archive_dir``, ``product_version``, ``data_source`, ``campaign_name``, ...]
    """
    # Retrieve path elements (os-specific)
    p = Path(path)
    list_path_elements = [str(part) for part in p.parts]
    # Retrieve where "DISDRODB" directory occurs
    idx_occurrence = np.where(np.isin(list_path_elements, "DISDRODB"))[0]
    # If DISDRODB directory not present, raise error
    if len(idx_occurrence) == 0:
        raise ValueError(f"The DISDRODB directory is not present in the path '{path}'")
    # Find the rightermost occurrence
    right_most_occurrence = max(idx_occurrence)
    # Define archive_dir and tree components
    archive_dir = os.path.join(*list_path_elements[: right_most_occurrence + 1])
    tree_components = list_path_elements[right_most_occurrence + 1 :]
    # Return components
    components = [archive_dir, *tree_components]
    return components


def infer_path_info_dict(path: str) -> dict:
    """Return a dictionary with the ``data_archive_dir``, ``data_source`` and ``campaign_name`` of the disdrodb_path.

    Parameters
    ----------
    path : str
        Directory or file path within the DISDRODB archive.

    Returns
    -------
    dict
        Dictionary with the path element of the DISDRODB archive.
        Valid keys: ``"data_archive_dir"``, ``"data_source"``, ``"campaign_name"``
    """
    components = infer_disdrodb_tree_path_components(path=path)
    if len(components) <= 3:
        raise ValueError(f"Impossible to determine data_source and campaign_name from {path}")
    path_dict = {}
    path_dict["data_archive_dir"] = components[0]
    path_dict["data_source"] = components[2]
    path_dict["campaign_name"] = components[3]
    return path_dict


def infer_path_info_tuple(path: str) -> tuple:
    """Return a tuple with the ``data_archive_dir``, ``data_source`` and ``campaign_name`` of the disdrodb_path.

    Parameters
    ----------
    path : str
        Directory or file path within the DISDRODB archive.

    Returns
    -------
    tuple
        Dictionary with the path element of the DISDRODB archive.
        Valid keys: ``"data_archive_dir"``, ``"data_source"``, ``"campaign_name"``
    """
    path_dict = infer_path_info_dict(path)
    return path_dict["data_archive_dir"], path_dict["data_source"], path_dict["campaign_name"]


def infer_disdrodb_tree_path(path: str) -> str:
    """Return the directory tree path from the archive directory.

    Current assumption: no ``data_source``, ``campaign_name``, ``station_name`` or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
        Directory or file path within the DISDRODB archive.

    Returns
    -------
    str
        Path inside the DISDRODB archive.
        Format: ``DISDRODB/RAW/<DATA_SOURCE>/<CAMPAIGN_NAME>/...``
        Format: ``DISDRODB/<ARCHIVE_VERSION>/<DATA_SOURCE>/<CAMPAIGN_NAME>/...``
    """
    components = infer_disdrodb_tree_path_components(path=path)
    tree_filepath = os.path.join("DISDRODB", *components[1:])
    return tree_filepath


def infer_archive_dir_from_path(path: str) -> str:
    """Return the disdrodb base directory from a file or directory path.

    Assumption: no data_source, campaign_name, station_name or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
        Directory or file path within the DISDRODB archive.

    Returns
    -------
    str
        Path of the DISDRODB directory.
    """
    return infer_disdrodb_tree_path_components(path=path)[0]


def infer_campaign_name_from_path(path: str) -> str:
    """Return the campaign name from a file or directory path.

    Assumption: no ``data_source``, ``campaign_name``, ``station_name`` or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
       Directory or file path within the DISDRODB archive.

    Returns
    -------
    str
        Name of the campaign.
    """
    components = infer_disdrodb_tree_path_components(path)
    if len(components) <= 3:
        raise ValueError(f"Impossible to determine campaign_name from {path}")
    campaign_name = components[3]
    return campaign_name


def infer_data_source_from_path(path: str) -> str:
    """Return the data_source from a file or directory path.

    Assumption: no ``data_source``, ``campaign_name``, ``station_name`` or file contain the word DISDRODB!

    Parameters
    ----------
    path : str
       Directory or file path within the DISDRODB archive.

    Returns
    -------
    str
        Name of the data source.
    """
    components = infer_disdrodb_tree_path_components(path)
    if len(components) <= 2:
        raise ValueError(f"Impossible to determine data_source from {path}")
    data_source = components[2]
    return data_source


####--------------------------------------------------------------------------.
#######################
#### Group utility ####
#######################


FILE_KEYS = [
    "product",
    "subproduct",
    "campaign_name",
    "station_name",
    "start_time",
    "end_time",
    "data_format",
    "accumulation_acronym",
    "sample_interval",
]


TIME_KEYS = [
    "year",
    "month",
    "month_name",
    "quarter",
    "season",
    "day",
    "doy",
    "dow",
    "hour",
    "minute",
    "second",
]


def check_groups(groups):
    """Check groups validity."""
    if not isinstance(groups, (str, list)):
        raise TypeError("'groups' must be a list (or a string if a single group is specified.")
    if isinstance(groups, str):
        groups = [groups]
    groups = np.array(groups)
    valid_keys = FILE_KEYS + TIME_KEYS
    invalid_keys = groups[np.isin(groups, valid_keys, invert=True)]
    if len(invalid_keys) > 0:
        raise ValueError(f"The following group keys are invalid: {invalid_keys}. Valid values are {valid_keys}.")
    return groups.tolist()


def get_season(time):
    """Get season from `datetime.datetime` or `datetime.date` object."""
    month = time.month
    if month in [12, 1, 2]:
        return "DJF"  # Winter (December, January, February)
    if month in [3, 4, 5]:
        return "MAM"  # Spring (March, April, May)
    if month in [6, 7, 8]:
        return "JJA"  # Summer (June, July, August)
    return "SON"  # Autumn (September, October, November)


def get_time_component(time, component):
    """Get time component from `datetime.datetime` object."""
    func_dict = {
        "year": lambda time: time.year,
        "month": lambda time: time.month,
        "day": lambda time: time.day,
        "doy": lambda time: time.timetuple().tm_yday,  # Day of year
        "dow": lambda time: time.weekday(),  # Day of week (0=Monday, 6=Sunday)
        "hour": lambda time: time.hour,
        "minute": lambda time: time.minute,
        "second": lambda time: time.second,
        # Additional
        "month_name": lambda time: time.strftime("%B"),  # Full month name
        "quarter": lambda time: (time.month - 1) // 3 + 1,  # Quarter (1-4)
        "season": lambda time: get_season(time),  # Season (DJF, MAM, JJA, SON)
    }
    return str(func_dict[component](time))


def _get_groups_value(groups, filepath):
    """Return the value associated to the groups keys.

    If multiple keys are specified, the value returned is a string of format: ``<group_value_1>/<group_value_2>/...``

    If a single key is specified and is ``start_time`` or ``end_time``, the function
    returns a :py:class:`datetime.datetime` object.
    """
    single_key = len(groups) == 1
    info_dict = get_info_from_filepath(filepath)
    start_time = info_dict["start_time"]
    list_key_values = []
    for key in groups:
        if key in TIME_KEYS:
            list_key_values.append(get_time_component(start_time, component=key))
        else:
            value = info_dict.get(key, f"{key}=None")
            list_key_values.append(value if single_key else str(value))
    if single_key:
        return list_key_values[0]
    return "/".join(list_key_values)


def group_filepaths(filepaths, groups=None):
    """
    Group filepaths in a dictionary if groups are specified.

    Parameters
    ----------
    filepaths : list
        List of filepaths.
    groups: list or str
        The group keys by which to group the filepaths.
        Valid group keys are ``product``, ``subproduct``, ``campaign_name``, ``station_name``,
        ``start_time``, ``end_time``,``accumulation_acronym``,``sample_interval``,
        ``data_format``,
        ``year``, ``month``, ``day``,  ``doy``, ``dow``, ``hour``, ``minute``, ``second``,
        ``month_name``, ``quarter``, ``season``.
        The time components are extracted from ``start_time`` !
        If groups is ``None`` returns the input filepaths list.
        The default value is ``None``.

    Returns
    -------
    dict or list
        Either a dictionary of format ``{<group_value>: <list_filepaths>}``.
        or the original input filepaths (if ``groups=None``)

    """
    if groups is None:
        return filepaths
    groups = check_groups(groups)
    filepaths_dict = defaultdict(list)
    _ = [filepaths_dict[_get_groups_value(groups, filepath)].append(filepath) for filepath in filepaths]
    return dict(filepaths_dict)
