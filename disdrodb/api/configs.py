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

from disdrodb.api.checks import check_product, check_sensor_name
from disdrodb.api.path import define_config_dir
from disdrodb.utils.yaml import read_yaml

logger = logging.getLogger(__name__)


def get_sensor_configs_dir(sensor_name: str, product: str) -> str:
    """Retrieve configs directory.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    product: str
        DISDRODB product.

    Returns
    -------
    config_sensor_dir: str
        Config directory.

    Raises
    ------
    ValueError
        Error if the config directory does not exist.
    """
    check_sensor_name(sensor_name)
    product = check_product(product)
    config_dir = define_config_dir(product=product)
    config_sensor_dir = os.path.join(config_dir, sensor_name)
    if not os.path.exists(config_sensor_dir):
        list_sensors = sorted(os.listdir(config_dir))
        print(f"Available sensor_name are {list_sensors}")
        raise ValueError(f"The config directory {config_sensor_dir} does not exist.")
    return config_sensor_dir


def read_config_file(sensor_name: str, product: str, filename: str) -> dict:
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
    check_sensor_name(sensor_name)
    product = check_product(product)
    config_sensor_dir = get_sensor_configs_dir(sensor_name, product=product)
    config_filepath = os.path.join(config_sensor_dir, filename)
    # Check yaml file exists
    if not os.path.exists(config_filepath):
        msg = f"{filename} not available in {config_sensor_dir}"
        logger.exception(msg)
        raise ValueError(msg)
    # Open dictionary
    dictionary = read_yaml(config_filepath)
    return dictionary


def available_sensor_names() -> list:
    """Get available names of sensors.

    Returns
    -------
    sensor_names: list
        Sorted list of the available sensors
    """
    config_dir = define_config_dir(product="L0A")
    return sorted(os.listdir(config_dir))
