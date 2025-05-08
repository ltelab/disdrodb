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
    r"""Initialize the DISDRODB directory structure for a station.

    It adds the relevant directories and the default issue and metadata YAML files..

    Parameters \n
    ---------- \n
    data_source : str \n
        Institution name (when campaign data spans more than 1 country), or country (when all campaigns (or sensor
        networks) are inside a given country).\n
        Must be UPPER CASE.\n
    campaign_name : str \n
        Campaign name. Must be UPPER CASE.\n
    station_name : str \n
        Station name \n
    data_archive_dir : str \n
        DISDRODB Data Archive directory \n
        Format: <...>/DISDRODB \n
        If not specified, uses path specified in the DISDRODB active configuration. \n
    """
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
