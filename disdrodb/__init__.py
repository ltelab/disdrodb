import contextlib
import importlib
import os
from importlib.metadata import PackageNotFoundError, version

from disdrodb._config import config  # noqa
from disdrodb.api.configs import available_sensor_names
from disdrodb.api.io import (
    available_campaigns,
    available_data_sources,
    available_stations,
)
from disdrodb.configs import define_disdrodb_configs as define_configs
from disdrodb.data_transfer.download_data import download_archive, download_station
from disdrodb.docs import open_documentation, open_sensor_documentation
from disdrodb.metadata import read_station_metadata
from disdrodb.metadata.checks import (
    check_archive_metadata_compliance,
    check_archive_metadata_geolocation,
)

PRODUCT_VERSION = "V0"
SOFTWARE_VERSION = "V" + importlib.metadata.version("disdrodb")
CONVENTIONS = "CF-1.10, ACDD-1.3"


__all__ = [
    "define_configs",
    "available_stations",
    "available_campaigns",
    "available_data_sources",
    "available_sensor_names",
    "check_archive_metadata_compliance",
    "check_archive_metadata_geolocation",
    "open_documentation",
    "open_sensor_documentation",
    "open_documentation",
    "read_station_metadata",
    "download_archive",
    "download_station",
]

__root_path__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Get version
with contextlib.suppress(PackageNotFoundError):
    __version__ = version("disdrodb")
