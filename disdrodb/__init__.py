import os
from importlib.metadata import PackageNotFoundError, version

from disdrodb._config import config
from disdrodb.api.io import (
    available_campaigns,
    available_data_sources,
    available_stations,
)
from disdrodb.api.metadata import read_station_metadata
from disdrodb.configs import define_disdrodb_configs as define_configs
from disdrodb.l0.standards import available_sensor_names

__root_path__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

__all__ = [
    "__root_path__",
    "config",
    "define_configs",
    "available_stations",
    "available_campaigns",
    "available_data_sources",
    "available_sensor_names",
    "read_station_metadata",
]

# Get version
try:
    __version__ = version("disdrodb")
except PackageNotFoundError:
    # package is not installed
    pass
