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
"""Open the documentation for the relevant sensor."""

import os
import webbrowser

from disdrodb.api.checks import check_sensor_name


def open_sensor_documentation(sensor_name):
    """Open the sensor documentation PDF in the browser."""
    from disdrodb import __root_path__

    check_sensor_name(sensor_name)
    docs_filepath = os.path.join(__root_path__, "disdrodb", "l0", "manuals", sensor_name + ".pdf")
    webbrowser.open(docs_filepath)


def open_documentation():
    """Open the DISDRODB documentation the browser."""
    docs_filepath = "https://disdrodb.readthedocs.io/en/latest/"
    webbrowser.open(docs_filepath)
