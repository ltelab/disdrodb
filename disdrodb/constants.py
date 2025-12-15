# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
METEOROLOGICAL_VARIABLES = ["air_temperature", "relative_humidity", "wind_speed", "wind_direction"]

OPTICAL_SENSORS = ["PARSIVEL", "PARSIVEL2", "LPM", "LPM_V0", "PWS100", "SWS250", "ODM470"]
IMPACT_SENSORS = ["RD80"]

PRODUCTS = ["RAW", "L0A", "L0B", "L0C", "L1", "L2E", "L2M"]

PRODUCTS_ARGUMENTS = {
    "L1": ["temporal_resolution"],
    "L2E": ["temporal_resolution"],
    "L2M": ["temporal_resolution", "model_name"],
}

PRODUCTS_REQUIREMENTS = {
    "L0A": "RAW",
    "L0B": "L0A",
    "L0C": "L0B",
    "L1": "L0C",
    "L2E": "L1",
    "L2M": "L2E",
}
