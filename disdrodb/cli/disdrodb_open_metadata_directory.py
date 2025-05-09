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
"""Routine to open the DISDRODB Data Archive logs directory."""

import sys
from typing import Optional

import click

from disdrodb.utils.cli import (
    click_metadata_archive_dir_option,
    click_station_arguments,
    parse_archive_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_station_arguments
@click_metadata_archive_dir_option
def disdrodb_open_metadata_directory(
    data_source: str,
    campaign_name: str,
    station_name: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Open the DISDRODB Data Archive logs directory of a station.

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
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``metadata_archive_dir`` path specified
        in the DISDRODB active configuration.
    """
    from disdrodb.api.io import open_metadata_directory

    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)

    open_metadata_directory(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
