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
"""Routine to download station data from the DISDRODB Decentralized Data Archive."""

import sys
from typing import Optional

import click

from disdrodb.data_transfer.download_data import click_download_options
from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    click_station_arguments,
    parse_archive_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_station_arguments
@click_data_archive_dir_option
@click_metadata_archive_dir_option
@click_download_options
def disdrodb_download_station(
    data_source: str,
    campaign_name: str,
    station_name: str,
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
    force: bool = False,
):
    """
    Download data of a single DISDRODB station from the DISDRODB remote repository.

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
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    force: bool, optional
        If ``True``, overwrite the already existing raw data file.
        The default value is ``False``.
    data_archive_dir : str (optional)
        DISDRODB Data Archive directory. Format: ``<...>/DISDRODB``.
        If ``None`` (the default), the disdrodb config variable ``data_archive_dir`` is used.
    """
    from disdrodb.data_transfer.download_data import download_station

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    download_station(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        force=force,
    )
