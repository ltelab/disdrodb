import os
from importlib.metadata import PackageNotFoundError, version

from disdrodb._config import config
from disdrodb.api.configs import available_sensor_names
from disdrodb.api.io import (
    available_campaigns,
    available_data_sources,
    available_stations,
)
from disdrodb.configs import define_disdrodb_configs as define_configs
from disdrodb.docs import open_documentation, open_sensor_documentation
from disdrodb.metadata import read_station_metadata
from disdrodb.metadata.check_metadata import (
    check_archive_metadata_compliance,
    check_archive_metadata_geolocation,
)

__root_path__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = [
    "__root_path__",
    "config",
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
]

# Get version
try:
    __version__ = version("disdrodb")
except PackageNotFoundError:
    # package is not installed
    pass
