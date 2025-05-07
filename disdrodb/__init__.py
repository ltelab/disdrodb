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
"""DISDRODB software."""

import contextlib
import importlib
import os
from importlib.metadata import PackageNotFoundError, version

from disdrodb._config import config  # noqa
from disdrodb.api.configs import available_sensor_names
from disdrodb.api.io import (
    find_files,
    open_dataset,
    open_logs_directory,
)
from disdrodb.api.search import (
    available_campaigns,
    available_data_sources,
    available_stations,
)
from disdrodb.configs import define_disdrodb_configs as define_configs
from disdrodb.configs import get_data_archive_dir, get_metadata_archive_dir
from disdrodb.data_transfer.download_data import download_archive, download_station
from disdrodb.docs import open_documentation, open_sensor_documentation
from disdrodb.l0.l0_reader import available_readers, get_reader, get_station_reader
from disdrodb.metadata import read_metadata_database, read_station_metadata
from disdrodb.metadata.checks import (
    check_archive_metadata_compliance,
    check_archive_metadata_geolocation,
)

ARCHIVE_VERSION = "V0"
SOFTWARE_VERSION = "V" + importlib.metadata.version("disdrodb")
CONVENTIONS = "CF-1.10, ACDD-1.3"

# Define coordinates names
# TODO: make it configurable
DIAMETER_COORDS = ["diameter_bin_center", "diameter_bin_width", "diameter_bin_lower", "diameter_bin_upper"]
VELOCITY_COORDS = ["velocity_bin_center", "velocity_bin_width", "velocity_bin_lower", "velocity_bin_upper"]
VELOCITY_DIMENSION = "velocity_bin_center"
DIAMETER_DIMENSION = "diameter_bin_center"
OPTICAL_SENSORS = ["OTT_Parsivel", "OTT_Parsivel2", "Thies_LPM"]
IMPACT_SENSORS = ["RD_80"]


PRODUCTS = ["RAW", "L0A", "L0B", "L0C", "L1", "L2E", "L2M"]


def available_products():
    """Return the list of available DISDRODB products."""
    return PRODUCTS


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

__all__ = [
    "define_configs",
    "available_stations",
    "available_campaigns",
    "available_data_sources",
    "available_sensor_names",
    "find_files",
    "get_reader",
    "get_station_reader",
    "get_metadata_archive_dir",
    "get_data_archive_dir",
    "available_readers",
    "open_dataset",
    "check_archive_metadata_compliance",
    "check_archive_metadata_geolocation",
    "open_documentation",
    "open_sensor_documentation",
    "open_logs_directory",
    "read_station_metadata",
    "read_metadata_database",
    "download_archive",
    "download_station",
]

__root_path__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def is_pytmatrix_available():
    """Check if the pytmatrix package is correctly installed and available."""
    try:
        import pytmatrix
    except Exception:
        return False
    return hasattr(pytmatrix, "psd")


# Get version
with contextlib.suppress(PackageNotFoundError):
    __version__ = version("disdrodb")
