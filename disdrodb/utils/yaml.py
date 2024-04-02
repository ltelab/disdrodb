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
"""YAML utility."""

import yaml


def read_yaml(filepath: str) -> dict:
    """Read a YAML file into a dictionary.

    Parameters
    ----------
    filepath : str
        Input YAML file path.

    Returns
    -------
    dict
        Dictionary with the attributes read from the YAML file.
    """
    with open(filepath) as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def write_yaml(dictionary, filepath, sort_keys=False):
    """Write a dictionary into a YAML file.

    Parameters
    ----------
    dictionary : dict
        Dictionary to write into a YAML file.
    """
    with open(filepath, "w") as f:
        yaml.dump(dictionary, f, sort_keys=sort_keys)
