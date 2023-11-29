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
"""Test Metadata Info Extraction."""
import os

from disdrodb.api.info import (
    infer_campaign_name_from_path,
    infer_data_source_from_path,
)
from disdrodb.configs import get_base_dir
from disdrodb.metadata.reader import read_station_metadata
from disdrodb.metadata.search import get_list_metadata


def get_archive_metadata_key_value(key: str, return_tuple: bool = True, base_dir: str = None):
    """Return the values of a metadata key for all the archive.

    Parameters
    ----------
    base_dir : str
        Path to the disdrodb directory.
    key : str
        Metadata key.
    return_tuple : bool, optional
       If True, returns a tuple of values with station, campaign and data source name.
       If False, returns a list of values without station, campaign and data source name.
       The default is True.
    base_dir : str (optional)
       Base directory of DISDRODB. Format: <...>/DISDRODB
       If None (the default), the disdrodb config variable 'dir' is used.

    Returns
    -------
    list or tuple
        List or tuple of values of the metadata key.
    """
    base_dir = get_base_dir(base_dir)
    list_metadata_paths = get_list_metadata(
        base_dir=base_dir, data_sources=None, campaign_names=None, station_names=None, with_stations_data=False
    )
    list_info = []
    for filepath in list_metadata_paths:
        data_source = infer_data_source_from_path(filepath)
        campaign_name = infer_campaign_name_from_path(filepath)
        station_name = os.path.basename(filepath).replace(".yml", "")
        metadata = read_station_metadata(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        value = metadata[key]
        info = (data_source, campaign_name, station_name, value)
        list_info.append(info)
    if not return_tuple:
        list_info = [info[3] for info in list_info]
    return list_info
