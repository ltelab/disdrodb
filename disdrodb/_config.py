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
"""DISDRODB donfig utility.

See https://donfig.readthedocs.io/en/latest/configuration.html for more info.
"""
from donfig import Config

from disdrodb.configs import read_disdrodb_configs


def _get_disdrodb_default_configs():
    """Retrieve the default DISDRODB settings from the ``.config_disdrodb.yml`` file."""
    try:
        config_dict = read_disdrodb_configs()
        config_dict = {key: value for key, value in config_dict.items() if value is not None}
    except Exception:
        config_dict = {}
    return config_dict


_CONFIG_DEFAULTS = {
    "data_archive_dir": None,
    "metadata_archive_dir": None,
    "zenodo_sandbox_token": None,
    "zenodo_token": None,
    "folder_partitioning": "year/month",
}
_CONFIG_DEFAULTS.update(_get_disdrodb_default_configs())

_CONFIG_PATHS = []

config = Config("disdrodb", defaults=[_CONFIG_DEFAULTS], paths=_CONFIG_PATHS)
