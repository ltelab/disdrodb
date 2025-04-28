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
"""Routines to read the DISDRODB Metadata."""

import pandas as pd

from disdrodb.api.path import define_metadata_filepath
from disdrodb.utils.yaml import read_yaml


def read_station_metadata(data_source, campaign_name, station_name, metadata_dir=None):
    """Open the station metadata YAML file into a dictionary.

    Parameters
    ----------
    data_source : str
        The name of the institution (for campaigns spanning multiple countries) or
        the name of the country (for campaigns or sensor networks within a single country).
        Must be provided in UPPER CASE.
    campaign_name : str
        The name of the campaign. Must be provided in UPPER CASE.
    station_name : str
        The name of the station.
    metadata_dir : str, optional
        The directory path where the DISDRODB Metadata Archive is located.
        If not specified, the path specified in the DISDRODB active configuration will be used.
        Expected path format: ``<...>/DISDRODB``.

    Returns
    -------
    metadata: dictionary
        The station metadata dictionary

    """
    # Retrieve metadata filepath
    metadata_filepath = define_metadata_filepath(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=True,
    )
    # Open the metadata file
    metadata_dict = read_yaml(metadata_filepath)
    return metadata_dict


def read_metadata_database(metadata_dir=None):
    """Read the metadata database."""
    from disdrodb.configs import get_metadata_dir
    from disdrodb.metadata.search import get_list_metadata

    metadata_dir = get_metadata_dir(metadata_dir=metadata_dir)

    list_metadata_paths = get_list_metadata(
        metadata_dir=metadata_dir,
        data_sources=None,
        campaign_names=None,
        station_names=None,
        with_stations_data=False,  # TODO: REFACTOR_STRUCTURE REMOVE !
    )
    list_metadata = [read_yaml(fpath) for fpath in list_metadata_paths]
    df = pd.DataFrame(list_metadata)
    return df
