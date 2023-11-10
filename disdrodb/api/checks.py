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
"""DISDRODB API Check Functions."""

import logging
import os
import re

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


def check_product(product):
    """Check DISDRODB product."""
    if not isinstance(product, str):
        raise TypeError("`product` must be a string.")
    valid_products = ["RAW", "L0A", "L0B"]
    if product.upper() not in valid_products:
        raise ValueError(f"Valid `products` are {valid_products}.")
    return product
