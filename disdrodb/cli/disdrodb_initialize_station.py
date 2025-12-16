# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
"""Script to initialize the DISDRODB station directory structure."""

import sys
from typing import Optional

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    click_station_arguments,
    parse_archive_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator


@click.command()
@click_station_arguments
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_initialize_station(
    # Station arguments
    data_source: str,
    campaign_name: str,
    station_name: str,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Initialize the DISDRODB directory structure for a new station.

    Creates the required directory structure and default YAML configuration files
    for a new station in both the DISDRODB Data and Metadata archives.

    \b
    Station Specification:
        Requires exact specification of data_source, campaign_name, and station_name.
        All three parameters must be provided and are case-sensitive (UPPER CASE required).

    \b
    Created Structure:
        Data Archive:
            - Data directory for the station where to place the raw data files

        Metadata Archive:
            - Station metadata YAML file (template)
            - Station issue YAML file (template)

    \b
    Archive Directories:
        --data_archive_dir: Custom path to DISDRODB data archive
        --metadata_archive_dir: Custom path to DISDRODB metadata archive
        If not specified, paths from the active DISDRODB configuration are used

    \b
    Examples:
        # Initialize a new station with default configuration
        disdrodb_initialize_station DATA_SOURCE CAMPAIGN_NAME STATION_NAME

        # Initialize with custom archive directories
        disdrodb_initialize_station DATA_SOURCE CAMPAIGN_NAME STATION_NAME --data_archive_dir /path/to/DISDRODB

    \b
    Important Notes:
        - Data source, campaign, and station names must be in UPPER CASE
        - All three station identifiers are required (no wildcards)
        - Creates template YAML files that need to be manually filled in
    """  # noqa: D301
    from disdrodb.api.create_directories import create_initial_station_structure

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)

    create_initial_station_structure(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Station arguments
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
