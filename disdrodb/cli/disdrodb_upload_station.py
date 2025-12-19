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
"""Routines to upload station data to the DISDRODB Decentralized Data Archive."""

import sys

import click

from disdrodb.data_transfer.upload_data import click_upload_options
from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    click_station_arguments,
    parse_archive_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_station_arguments
@click_upload_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_upload_station(
    # Station arguments
    data_source: str,
    campaign_name: str,
    station_name: str,
    # Upload options
    platform: str | None = None,
    force: bool = False,
    # DISDRODB root directories
    data_archive_dir: str | None = None,
    metadata_archive_dir: str | None = None,
):
    """Upload raw data from a single DISDRODB station to the DISDRODB Decentralized Data Archive.

    Currently, only upload to the Zenodo data repository is implemented.

    The station metadata file is automatically updated with the remote data URL.

    PLEASE UPLOAD JUST YOUR DATA !


    \b
    Upload Behavior:
        By default, upload is skipped if data already exists on a remote location.
        Use '--force True' to upload even if data already exists remotely.
        Warning: Forcing upload will overwrite the existing disdrodb_data_url in metadata.

    \b
    Upload Options:
        --platform: Remote platform name (default: 'sandbox.zenodo' for testing)
            Use 'zenodo' for final data dissemination
        --force: Upload even if data already exists remotely (default: False)

    \b
    Archive Directories:
        --data_archive_dir: Custom path to DISDRODB data archive
        --metadata_archive_dir: Custom path to DISDRODB metadata archive
        If not specified, paths from the active DISDRODB configuration are used

    \b
    Examples:
        # Upload a station to sandbox for testing
        disdrodb_upload_station DATA_SOURCE CAMPAIGN_NAME STATION_NAME

        # Upload a station to production Zenodo
        disdrodb_upload_station DATA_SOURCE CAMPAIGN_NAME STATION_NAME --platform zenodo

        # Force upload and update metadata URL
        disdrodb_upload_station DATA_SOURCE CAMPAIGN_NAME STATION_NAME --force True
    """  # noqa: D301
    from disdrodb.data_transfer.upload_data import upload_station

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    upload_station(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Station argument
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Upload options
        platform=platform,
        force=force,
    )
