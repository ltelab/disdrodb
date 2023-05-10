from importlib.metadata import PackageNotFoundError, version

from disdrodb.api.io import (
    available_campaigns,
    available_data_sources,
    available_stations,
)
from disdrodb.api.metadata import read_station_metadata
from disdrodb.l0.standards import available_sensor_name

__all__ = [
    "available_stations",
    "available_campaigns",
    "available_data_sources",
    "available_sensor_name",
    "read_station_metadata",
]

# Get version
try:
    __version__ = version("disdrodb")
except PackageNotFoundError:
    # package is not installed
    pass
