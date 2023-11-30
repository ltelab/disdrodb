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
"""DISDRODB Checks Functions."""

import logging
import os
import re

from disdrodb.api.info import infer_disdrodb_tree_path_components
from disdrodb.api.path import (
    define_issue_dir,
    define_issue_filepath,
    define_metadata_dir,
    define_metadata_filepath,
    define_station_dir,
)
from disdrodb.utils.directories import (
    ensure_string_path,
    list_files,
    remove_path_trailing_slash,
)

logger = logging.getLogger(__name__)


def check_path(path: str) -> None:
    """Check if a path exists.

    Parameters
    ----------
    path : str
        Path to check.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist. Please check the path.")


def check_url(url: str) -> bool:
    """Check url.

    Parameters
    ----------
    url : str
        URL to check.

    Returns
    -------
    bool
        True if url well formatted, False if not well formatted.
    """
    regex = r"^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"  # noqa: E501

    if re.match(regex, url):
        return True
    else:
        return False


def check_path_is_a_directory(dir_path, path_name=""):
    """Check that the path exists and is directory."""
    dir_path = ensure_string_path(dir_path, msg="Provide {path_name} as a string", accepth_pathlib=True)
    if not os.path.exists(dir_path):
        raise ValueError(f"{path_name} {dir_path} directory does not exist.")
    if not os.path.isdir(dir_path):
        raise ValueError(f"{path_name} {dir_path} is not a directory.")


def check_directories_inside(dir_path):
    """Check there are directories inside the specified dir_path."""
    dir_paths = os.listdir(dir_path)
    if len(dir_paths) == 0:
        raise ValueError(f"There are not directories within {dir_path}")


def check_base_dir(base_dir: str):
    """Raise an error if the path does not end with "DISDRODB"."""
    base_dir = str(base_dir)  # convert Pathlib to string
    if not base_dir.endswith("DISDRODB"):
        raise ValueError(f"The path {base_dir} does not end with DISDRODB. Please check the path.")
    return base_dir


def check_sensor_name(sensor_name: str, product: str = "L0A") -> None:
    """Check sensor name.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    product : str
        DISDRODB product.

    Raises
    ------
    TypeError
        Error if `sensor_name` is not a string.
    ValueError
        Error if the input sensor name has not been found in the list of available sensors.
    """
    from disdrodb.api.configs import available_sensor_names

    sensor_names = available_sensor_names(product=product)
    if not isinstance(sensor_name, str):
        raise TypeError("'sensor_name' must be a string.")
    if sensor_name not in sensor_names:
        msg = f"{sensor_name} not valid {sensor_name}. Valid values are {sensor_names}."
        logger.error(msg)
        raise ValueError(msg)


def check_campaign_name(campaign_name):
    """Check the campaign name is upper case !"""
    upper_campaign_name = campaign_name.upper()
    if campaign_name != upper_campaign_name:
        msg = f"The campaign directory name {campaign_name} must be defined uppercase: {upper_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)


def check_data_source(data_source):
    """Check the data_source name is upper case !"""
    upper_data_source = data_source.upper()
    if data_source != upper_data_source:
        msg = f"The data source directory name {data_source} must be defined uppercase: {upper_data_source}"
        logger.error(msg)
        raise ValueError(msg)


def check_product(product):
    """Check DISDRODB product."""
    if not isinstance(product, str):
        raise TypeError("`product` must be a string.")
    valid_products = ["RAW", "L0A", "L0B"]
    if product.upper() not in valid_products:
        msg = f"Valid `products` are {valid_products}."
        logger.error(msg)
        raise ValueError(msg)
    return product


def check_station_dir(product, data_source, campaign_name, station_name, base_dir=None):
    """Check existence of the station data directory. If does not exists, raise an error."""
    station_dir = define_station_dir(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    if not os.path.exists(station_dir) and os.path.isdir(station_dir):
        msg = f"The station {station_name} data directory does not exist at {station_dir}."
        logger.error(msg)
        raise ValueError(msg)
    return station_dir


def has_available_station_files(product, data_source, campaign_name, station_name, base_dir=None):
    """Return True if data are available for the given product and station."""
    station_dir = check_station_dir(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    filepaths = list_files(station_dir, glob_pattern="*", recursive=True)
    nfiles = len(filepaths)
    return nfiles >= 1


def check_station_has_data(product, data_source, campaign_name, station_name, base_dir=None):
    """Check the station data directory has data inside. If not, raise an error."""
    if not has_available_station_files(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    ):
        msg = f"The {product} station data directory of {data_source} {campaign_name} {station_name} is empty !"
        logger.error(msg)
        raise ValueError(msg)


def check_metadata_dir(product, data_source, campaign_name, base_dir=None):
    """Check existence of the metadata directory. If does not exists, raise an error."""
    metadata_dir = define_metadata_dir(
        product=product, base_dir=base_dir, data_source=data_source, campaign_name=campaign_name, check_exists=False
    )
    if not os.path.exists(metadata_dir) and os.path.isdir(metadata_dir):
        msg = f"The metadata directory does not exist at {metadata_dir}."
        logger.error(msg)
        raise ValueError(msg)
    return metadata_dir


def check_metadata_file(product, data_source, campaign_name, station_name, base_dir=None, check_validity=True):
    """Check existence of a valid metadata YAML file. If does not exists, raise an error."""
    from disdrodb.metadata.checks import check_metadata_compliance

    _ = check_metadata_dir(product=product, base_dir=base_dir, data_source=data_source, campaign_name=campaign_name)
    metadata_filepath = define_metadata_filepath(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    # Check existence
    if not os.path.exists(metadata_filepath):
        msg = (
            f"The metadata YAML file of {data_source} {campaign_name} {station_name} does not exist at"
            f" {metadata_filepath}."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Check validity
    if check_validity:
        check_metadata_compliance(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
        )
    return metadata_filepath


def check_issue_dir(data_source, campaign_name, base_dir=None):
    """Check existence of the issue directory. If does not exists, raise an error."""
    issue_dir = define_issue_dir(
        base_dir=base_dir, data_source=data_source, campaign_name=campaign_name, check_exists=False
    )
    if not os.path.exists(issue_dir) and os.path.isdir(issue_dir):
        msg = "The issue directory does not exist at {issue_dir}."
        logger.error(msg)
        raise ValueError(msg)
    return issue_dir


def check_issue_file(data_source, campaign_name, station_name, base_dir=None):
    """Check existence of a valid issue YAML file. If does not exists, raise an error."""
    from disdrodb.issue.checks import check_issue_compliance

    _ = check_issue_dir(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    issue_filepath = define_issue_filepath(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    # Check existence
    if not os.path.exists(issue_filepath):
        msg = f"The issue YAML file of {data_source} {campaign_name} {station_name} does not exist at {issue_filepath}."
        logger.error(msg)
        raise ValueError(msg)

    # Check validity
    check_issue_compliance(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    return issue_filepath


def check_is_within_raw_directory(path):
    """Check the path is within the DISDRODB 'Raw' directory."""
    components = infer_disdrodb_tree_path_components(path)
    if components[1] != "Raw":
        msg = f"{path} is not within the 'Raw' directory."
        logger.error(msg)
        raise ValueError(msg)


def check_is_within_processed_directory(path):
    """Check the path is within the DISDRODB 'Processed' directory."""
    components = infer_disdrodb_tree_path_components(path)
    if components[1] != "Processed":
        msg = f"{path} is not within the 'Processed' directory."
        logger.error(msg)
        raise ValueError(msg)


def check_valid_campaign_dir(campaign_dir):
    """Check the validity of a campaign directory path.

    Used to check validity of 'raw_dir' and 'processed_dir'.

    The path must represents this path */DISDRODB/<Raw or Processed>/<DATA_SOURCE>/<CAMPAIGN_NAME>
    """
    last_component = os.path.basename(campaign_dir)
    tree_components = infer_disdrodb_tree_path_components(campaign_dir)
    tree_path = "/".join(tree_components)
    # Check that is not data_source or 'Raw'/Processed' directory
    if len(tree_components) < 4:
        msg = (
            "Expecting the campaign directory path to comply with the pattern <...>/DISDRODB//<Raw or"
            " Processed>/<DATA_SOURCE>/<CAMPAIGN_NAME>."
        )
        msg = msg + f"It only provides {tree_path}"
        logger.error(msg)
        raise ValueError(msg)
    # Check that ends with the campaign_name
    campaign_name = tree_components[3]
    if last_component != campaign_name:
        msg = (
            "Expecting the campaign directory path to comply with the pattern  <...>/DISDRODB//<Raw or"
            " Processed>/<DATA_SOURCE>/<CAMPAIGN_NAME>."
        )
        msg = msg + f"The 'campaign directory path {campaign_dir} does not end with '{campaign_name}'!"
        logger.error(msg)
        raise ValueError(msg)


def check_raw_dir(raw_dir: str, station_name: str) -> None:
    """Check validity of raw_dir content.

    Steps:
    1. Check that 'raw_dir' is a valid directory path
    2. Check that 'raw_dir' follows the expect directory structure
    3. Check that each station_name directory contains data
    4. Check that for each station_name the mandatory metadata.yml is specified.
    5. Check that for each station_name the mandatory issue.yml is specified.

    Parameters
    ----------
    raw_dir : str
        Input raw campaign directory.
    station_name : str
        Station name.
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.

    """
    # Ensure valid path format
    raw_dir = remove_path_trailing_slash(raw_dir)

    # Check raw_dir is an existing directory
    check_path_is_a_directory(raw_dir, path_name="raw_dir")

    # Check is a valid campaign directory path
    check_valid_campaign_dir(raw_dir)

    # Check is inside the 'Raw' directory
    check_is_within_raw_directory(raw_dir)

    # Retrieve data_source and campaign_name
    base_dir, product_type, data_source, campaign_name = infer_disdrodb_tree_path_components(raw_dir)

    # Check <DATA_SOURCE> and <CAMPAIGN_NAME> are upper case
    check_campaign_name(campaign_name)
    check_data_source(data_source)

    # Check there are directories in raw_dir
    check_directories_inside(raw_dir)

    # Check there is data in the station directory
    check_station_has_data(
        product="RAW",
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check there is a valid metadata YAML file
    check_metadata_file(
        product="RAW",
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check there is valid issue YAML file
    check_issue_file(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    return raw_dir


def check_processed_dir(processed_dir):
    """Check input, format and validity of the 'processed_dir' directory path.

    Parameters
    ----------
    processed_dir : str
        Path to the campaign directory in the 'DISDRODB/Processed directory tree

    Returns
    -------
    str
        Path of the processed campaign directory
    """
    # Check path type
    processed_dir = ensure_string_path(processed_dir, msg="Provide 'processed_dir' as a string", accepth_pathlib=True)

    # Ensure valid path format
    processed_dir = remove_path_trailing_slash(processed_dir)

    # Check is a valid campaign directory path
    # - <...>/DISDRODB/Processed/<DATA_SOURCE>/<CAMPAIGN_NAME>
    check_valid_campaign_dir(processed_dir)

    # Check is inside the 'Processed' directory
    check_is_within_processed_directory(processed_dir)

    # Retrieve data_source and campaign_name
    base_dir, product_type, data_source, campaign_name = infer_disdrodb_tree_path_components(processed_dir)

    # Check <DATA_SOURCE> and <CAMPAIGN_NAME> are upper case
    check_campaign_name(campaign_name)
    check_data_source(data_source)

    return processed_dir
