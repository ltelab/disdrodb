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


def read_station_metadata(data_source, campaign_name, station_name, metadata_archive_dir=None):
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
    metadata_archive_dir : str, optional
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
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=True,
    )
    # Open the metadata file
    metadata_dict = read_yaml(metadata_filepath)
    return metadata_dict


def read_metadata_archive(
    metadata_archive_dir=None,
    data_sources=None,
    campaign_names=None,
    station_names=None,
    available_data=False,
):
    """Read the DISDRODB Metadata Archive Database.

    Parameters
    ----------
    metadata_archive_dir : str or Path-like, optional
        Path to the root of the DISDRODB Metadata Archive. If None, the
        default metadata base directory is used. Default is None.
    data_sources : str or sequence of str, optional
        One or more data source identifiers to filter stations by. If None,
        no filtering on data source is applied. The default is is None.
    campaign_names : str or sequence of str, optional
        One or more campaign names to filter stations by. If None, no filtering
        on campaign is applied. The default is is None.
    station_names : str or sequence of str, optional
        One or more station names to include. If None, all stations matching
        other filters are considered. The default is is None.
    available_data: bool, optional
        If True, only information of stations with data available in the online
        DISDRODB Decentralized Data Archive are returned.
        If False (the default), all stations present in the DISDRODB Metadata Archive
        matching the filtering criteria are returned,

    Returns
    -------
    pandas.DataFrame

    """
    from disdrodb.configs import get_metadata_archive_dir
    from disdrodb.metadata.search import get_list_metadata

    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir=metadata_archive_dir)

    list_metadata_paths = get_list_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        product=None,  # --> Search in DISDRODB Metadata Archive
        available_data=available_data,
    )
    list_metadata = [read_yaml(fpath) for fpath in list_metadata_paths]
    df = pd.DataFrame(list_metadata)
    return df
