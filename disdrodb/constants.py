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
"""DISDRODB constants."""
import importlib

ARCHIVE_VERSION = "V0"
SOFTWARE_VERSION = "V" + importlib.metadata.version("disdrodb")
CONVENTIONS = "CF-1.10, ACDD-1.3"

# Define coordinates names
DIAMETER_COORDS = ["diameter_bin_center", "diameter_bin_width", "diameter_bin_lower", "diameter_bin_upper"]
VELOCITY_COORDS = ["velocity_bin_center", "velocity_bin_width", "velocity_bin_lower", "velocity_bin_upper"]
GEOLOCATION_COORDS = ["longitude", "latitude", "altitude"]
VELOCITY_DIMENSION = "velocity_bin_center"
DIAMETER_DIMENSION = "diameter_bin_center"
COORDINATES = [
    "diameter_bin_center",
    "diameter_bin_width",
    "diameter_bin_upper",
    "velocity_bin_lower",
    "velocity_bin_center",
    "velocity_bin_width",
    "velocity_bin_upper",
    "latitude",
    "longitude",
    "altitude",
    "time",
    "sample_interval",
]
OPTICAL_SENSORS = ["PARSIVEL", "PARSIVEL2", "LPM", "PWS100"]
IMPACT_SENSORS = ["RD80"]

PRODUCTS = ["RAW", "L0A", "L0B", "L0C", "L1", "L2E", "L2M"]

PRODUCTS_ARGUMENTS = {
    "L2E": ["rolling", "sample_interval"],
    "L2M": ["rolling", "sample_interval", "model_name"],
}

PRODUCTS_REQUIREMENTS = {
    "L0A": "RAW",
    "L0B": "L0A",
    "L0C": "L0B",
    "L1": "L0C",
    "L2E": "L1",
    "L2M": "L2E",
}