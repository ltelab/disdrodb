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
    """Download raw data of a single station from the DISDRODB Decentralized Data Archive.

    It downloads station raw data files and stores them in the local DISDRODB Data Archive.
    The data are organized by data_source, campaign_name, and station_name.

    For stations data hosted on FTP/webservers, recursive calls of
    this command allows to fetch and download just the new data when becomes available.

    For stations data hosted on data repository such as Zenodo in ZIP archives,
    if a new version becomes available, you must set force=True to download the new version.

    \b
    Station Specification:
        Requires exact specification of data_source, campaign_name, and station_name.
        All three parameters must be provided and are case-sensitive (UPPER CASE required).

    \b
    Download Behavior:
        For webserver/FTP-hosted data:
            - Incremental downloads: Fetch only new files when they become available
            - Existing files on disk are skipped unless --force is used

        For repository-hosted data (e.g., Zenodo):
            - Use --force to download new versions when available
            - Without --force, download is skipped if data already exists locally

    \b
    Download Options:
        --force: Removes existing raw data files and forces complete re-download (default: False)
        WARNING: All existing station data will be deleted before re-downloading.

    \b
    Archive Directories:
        --data_archive_dir: Custom path to DISDRODB data archive
        --metadata_archive_dir: Custom path to DISDRODB metadata archive
        If not specified, paths from the active DISDRODB configuration are used

    \b
    Examples:
        # Download data for a single station
        disdrodb_download_station EPFL HYMEX_LTE_SOP2 10

        # Force re-download of existing data
        disdrodb_download_station EPFL HYMEX_LTE_SOP2 10 --force True

        # Download with custom archive directory
        disdrodb_download_station NASA IFLOODS apu01 --data_archive_dir /path/to/DISDRODB

    \b
    Important Notes:
        - Data source, campaign, and station names must be UPPER CASE
        - All three station identifiers are required (no wildcards)
        - Downloaded files are placed in <data_archive_dir>/Raw/<data_source>/<campaign_name>/<station_name>/
        - Use --force with caution as it will remove data on disk before starting to re-download them.
    """  # noqa: D301
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
