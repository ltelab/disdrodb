from importlib.metadata import version, PackageNotFoundError

from disdrodb.api.io import (
    available_stations,
    available_campaigns,
    available_data_sources,
)
from disdrodb.api.metadata import read_station_metadata

__all__ = [
    "available_stations",
    "available_campaigns",
    "available_data_sources",
    "read_station_metadata",
]

# Get version
try:
    __version__ = version("disdrodb")
except PackageNotFoundError:
    # package is not installed
    pass
