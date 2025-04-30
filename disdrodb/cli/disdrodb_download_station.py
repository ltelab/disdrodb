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
"""Routines to download station data from the DISDRODB Decentralized Data Archive."""

import sys
from typing import Optional

import click

from disdrodb.data_transfer.download_data import click_download_options
from disdrodb.utils.cli import (
    click_base_dir_option,
    click_metadata_dir_option,
    click_station_arguments,
    parse_root_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_station_arguments
@click_base_dir_option
@click_metadata_dir_option
@click_download_options
def disdrodb_download_station(
    data_source: str,
    campaign_name: str,
    station_name: str,
    base_dir: Optional[str] = None,
    metadata_dir: Optional[str] = None,
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
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    force: bool, optional
        If ``True``, overwrite the already existing raw data file.
        The default is ``False``.
    base_dir : str (optional)
        Base directory of DISDRODB. Format: ``<...>/DISDRODB``.
        If ``None`` (the default), the disdrodb config variable ``base_dir`` is used.
    """
    from disdrodb.data_transfer.download_data import download_station

    base_dir = parse_root_dir(base_dir)
    metadata_dir = parse_root_dir(metadata_dir)
    download_station(
        base_dir=base_dir,
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        force=force,
    )
