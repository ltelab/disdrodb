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
"""DISDRODB software."""

import contextlib
import os
from importlib.metadata import PackageNotFoundError, version

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import disdrodb.accessor  # noqa
from disdrodb._config import config  # noqa
from disdrodb.api.configs import available_sensor_names
from disdrodb.api.io import (
    find_files,
    open_dataset,
    open_logs_directory,
    open_metadata_directory,
    open_product_directory,
)
from disdrodb.api.search import (
    available_campaigns,
    available_data_sources,
    available_stations,
)
from disdrodb.configs import (
    define_configs,
    get_data_archive_dir,
    get_metadata_archive_dir,
    get_scattering_table_dir,
)
from disdrodb.data_transfer.download_data import download_archive, download_station
from disdrodb.docs import open_documentation, open_sensor_documentation
from disdrodb.l0 import (
    available_readers,
    generate_l0a,
    generate_l0b,
    generate_l0b_from_nc,
    get_reader,
    get_station_reader,
)
from disdrodb.l1 import generate_l1
from disdrodb.l2 import generate_l2_radar, generate_l2e, generate_l2m
from disdrodb.metadata import download_metadata_archive, read_metadata_archive, read_station_metadata
from disdrodb.metadata.checks import (
    check_metadata_archive,
    check_metadata_archive_geolocation,
    check_station_metadata,
)
from disdrodb.routines import (
    create_summary,
    create_summary_station,
    run_l0,
    run_l0_station,
    run_l0a,
    run_l0a_station,
    run_l0b,
    run_l0b_station,
    run_l0c,
    run_l0c_station,
    run_l1,
    run_l1_station,
    run_l2e,
    run_l2e_station,
    run_l2m,
    run_l2m_station,
    run,
    run_station,
)
from disdrodb.utils.manipulations import convert_from_decibel as idecibel
from disdrodb.utils.manipulations import convert_to_decibel as decibel

from .constants import (
    ARCHIVE_VERSION,
    CONVENTIONS,
    COORDINATES,
    DIAMETER_COORDS,
    DIAMETER_DIMENSION,
    GEOLOCATION_COORDS,
    IMPACT_SENSORS,
    OPTICAL_SENSORS,
    PRODUCTS,
    PRODUCTS_ARGUMENTS,
    PRODUCTS_REQUIREMENTS,
    SOFTWARE_VERSION,
    VELOCITY_COORDS,
    VELOCITY_DIMENSION,
)


def available_products():
    """Return the list of available DISDRODB products."""
    return PRODUCTS


__all__ = [
    "ARCHIVE_VERSION",
    "CONVENTIONS",
    "COORDINATES",
    "DIAMETER_COORDS",
    "DIAMETER_DIMENSION",
    "GEOLOCATION_COORDS",
    "IMPACT_SENSORS",
    "OPTICAL_SENSORS",
    "PRODUCTS",
    "PRODUCTS_ARGUMENTS",
    "PRODUCTS_REQUIREMENTS",
    "SOFTWARE_VERSION",
    "VELOCITY_COORDS",
    "VELOCITY_DIMENSION",
    "available_campaigns",
    "available_data_sources",
    "available_readers",
    "available_sensor_names",
    "available_stations",
    "check_metadata_archive",
    "check_metadata_archive_geolocation",
    "check_station_metadata",
    "create_summary",
    "create_summary_station",
    "decibel",
    "define_configs",
    "download_archive",
    "download_metadata_archive",
    "download_station",
    "find_files",
    "generate_l0a",
    "generate_l0b",
    "generate_l0b_from_nc",
    "generate_l1",
    "generate_l2_radar",
    "generate_l2e",
    "generate_l2m",
    "get_data_archive_dir",
    "get_metadata_archive_dir",
    "get_reader",
    "get_scattering_table_dir",
    "get_station_reader",
    "idecibel",
    "open_dataset",
    "open_documentation",
    "open_logs_directory",
    "open_metadata_directory",
    "open_product_directory",
    "open_sensor_documentation",
    "read_metadata_archive",
    "read_station_metadata",
    # Functions invoking the disdrodb_run_* scripts in the terminals
    "run",
    "run_l0",
    "run_l0_station",
    "run_l0a",
    "run_l0a_station",
    "run_l0b",
    "run_l0b_station",
    "run_l0c",
    "run_l0c_station",
    "run_l1",
    "run_l1_station",
    "run_l2e",
    "run_l2e_station",
    "run_l2m",
    "run_l2m_station",
    "run_station",
]


package_dir = os.path.dirname(os.path.realpath(__file__))


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
