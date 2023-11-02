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


def _try_get_default_disdrodb_dir():
    """Retrieve the default DISDRODB directory specified in the .config_disdrodb.yml file."""
    try:
        disdrodb_dir = read_disdrodb_configs().get("disdrodb_dir", None)
    except Exception:
        disdrodb_dir = None
    return disdrodb_dir


_CONFIG_DEFAULTS = {
    "dir": _try_get_default_disdrodb_dir(),  # DISDRODB_DIR
}

_CONFIG_PATHS = []

config = Config("disdrodb", defaults=[_CONFIG_DEFAULTS], paths=_CONFIG_PATHS)
