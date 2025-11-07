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
    """Open the DISDRODB Metadata Archive directory of a station in the system file explorer.

    The command allows to easily browse stations metadata YAML files.

    \b
    Station Specification:
        Requires data_source and campaign_name (UPPER CASE required).
        station_name is optional.

    \b
    Archive Directory:
        --metadata_archive_dir: Custom path to DISDRODB Metadata Archive
        If not specified, the path from the active DISDRODB configuration is used

    \b
    Examples:
        # Open metadata directory for a specific station
        disdrodb_open_metadata_directory EPFL HYMEX_LTE_SOP2 10

        # Open metadata directory for a specific station
        disdrodb_open_metadata_directory NASA IFLOODS

        # Open with custom metadata archive directory
        disdrodb_open_metadata_directory EPFL HYMEX_LTE_SOP2 --metadata_archive_dir /path/to/DISDRODB-METADATA/DISDRODB

    \b
    Important Notes:
        - Data source and campaign names must be in UPPER CASE
        - Opens the directory in your system's default file manager
        - Useful for manual inspection and editing of station metadata YAML files
        - If station_name is omitted, opens the campaign-level metadata directory
    """  # noqa: D301
    from disdrodb.api.io import open_metadata_directory

    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)

    open_metadata_directory(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
