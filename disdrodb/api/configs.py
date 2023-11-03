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
"""Retrieve sensor configuration files."""

import logging
import os

from disdrodb.api.checks import check_product_level, check_sensor_name
from disdrodb.utils.yaml import read_yaml

logger = logging.getLogger(__name__)


def _get_config_dir(product_level):
    """Define the config directory path of a given DISDRODB product level."""
    from disdrodb import __root_path__

    product_level = product_level.lower()
    config_dir_path = os.path.join(__root_path__, "disdrodb", product_level, "configs")
    return config_dir_path


def get_sensor_configs_dir(sensor_name: str, product_level: str) -> str:
    """Retrieve configs directory.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    product_level: str
        DISDRODB product level.

    Returns
    -------
    config_sensor_dir: str
        Config directory.

    Raises
    ------
    ValueError
        Error if the config directory does not exist.
    """
    check_sensor_name(sensor_name, product_level=product_level)
    product_level = check_product_level(product_level)
    config_dir_path = _get_config_dir(product_level=product_level)
    config_sensor_dir_path = os.path.join(config_dir_path, sensor_name)
    if not os.path.exists(config_sensor_dir_path):
        list_sensors = sorted(os.listdir(config_dir_path))
        print(f"Available sensor_name are {list_sensors}")
        raise ValueError(f"The config directory {config_sensor_dir_path} does not exist.")
    return config_sensor_dir_path


def read_config_file(sensor_name: str, product_level: str, filename: str) -> dict:
    """Read a config yaml file and return the dictionary.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    filename : str
        Name of the file.

    Returns
    -------
    dict
        Content of the config file.

    Raises
    ------
    ValueError
        Error if file does not exist.
    """
    check_sensor_name(sensor_name, product_level=product_level)
    product_level = check_product_level(product_level)
    config_sensor_dir_path = get_sensor_configs_dir(sensor_name, product_level=product_level)
    config_fpath = os.path.join(config_sensor_dir_path, filename)
    # Check yaml file exists
    if not os.path.exists(config_fpath):
        msg = f"{filename} not available in {config_sensor_dir_path}"
        logger.exception(msg)
        raise ValueError(msg)
    # Open dictionary
    dictionary = read_yaml(config_fpath)
    return dictionary


def available_sensor_names(product_level: str = "L0") -> sorted:
    """Get available names of sensors.

    Returns
    -------
    sensor_names: list
        Sorted list of the available sensors
    product_level: str
        DISDRODB product level.
        By default, it returns the sensors available for DISDRODB L0 products.
    """
    product_level = check_product_level(product_level)
    config_dir_path = _get_config_dir(product_level=product_level)
    return sorted(os.listdir(config_dir_path))
