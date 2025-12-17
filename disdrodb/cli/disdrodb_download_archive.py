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
"""Wrapper to download stations from the DISDRODB Decentralized Data Archive."""

import sys

import click

from disdrodb.data_transfer.download_data import click_download_archive_options, click_download_options
from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    parse_archive_dir,
    parse_arg_to_list,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_download_archive_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
@click_download_options
def disdrodb_download_archive(
    data_sources: str | None = None,
    campaign_names: str | None = None,
    station_names: str | None = None,
    data_archive_dir: str | None = None,
    metadata_archive_dir: str | None = None,
    force: bool = False,
):
    """Download raw data for multiple DISDRODB stations from the DISDRODB Decentralized Data Archive.

    It downloads station raw data files and stores them in the local DISDRODB Data Archive.
    The data are organized by data_source, campaign_name, and station_name.

    \b
    Station Selection:
        If no station filters are specified, downloads ALL stations.
        Use data_sources, campaign_names, and station_names to filter stations.
        Filters work together to narrow down the selection (AND logic).
        Only stations with a ``disdrodb_data_url`` in their metadata will be downloaded.

    \b
    Download Behavior:
        For webserver/FTP-hosted data:
            - Incremental downloads: Fetch only new files when they become available
            - Existing files on disk are skipped unless '--force True' is used

        For repository-hosted data (e.g., Zenodo):
            - Use '--force True' to download new versions when available
            - Without '--force True', download is skipped if data already exists locally

    \b
    Download Options:
        --force: Removes existing raw data files and forces complete re-download (default: False)
        Warning: All existing station data will be deleted before re-downloading

    \b
    Archive Directories:
        --data_archive_dir: Custom path to DISDRODB data archive
        --metadata_archive_dir: Custom path to DISDRODB metadata archive
        If not specified, paths from the active DISDRODB configuration are used

    \b
    Examples:
        # Download all stations with available download URLs
        disdrodb_download_archive

        # Download all stations from specific data sources
        disdrodb_download_archive --data_sources 'EPFL NASA'

        # Download specific campaigns and force re-download
        disdrodb_download_archive --campaign_names 'HYMEX_LTE_SOP2 IFLOODS' --force True
    """  # noqa: D301
    from disdrodb.data_transfer.download_data import download_archive

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    download_archive(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        force=force,
    )
