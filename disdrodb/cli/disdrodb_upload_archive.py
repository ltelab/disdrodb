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
"""Routines to upload data to the DISDRODB Decentralized Data Archive."""

import sys
from typing import Optional

import click

from disdrodb.data_transfer.upload_data import click_upload_archive_options, click_upload_options
from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    parse_archive_dir,
    parse_arg_to_list,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_upload_archive_options
@click_upload_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_upload_archive(
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
    # Stations options
    data_sources: Optional[str] = None,
    campaign_names: Optional[str] = None,
    station_names: Optional[str] = None,
    # Upload options
    platform: Optional[str] = None,
    force: bool = False,
):
    """Upload raw data for multiple DISDRODB stations to the DISDRODB Decentralized Data Archive.

    Currently, only upload to the Zenodo data repository is implemented.
    The station metadata files are automatically updated with the remote data URL.

    PLEASE UPLOAD JUST YOUR DATA !

    \b
    Station Selection:
        If no station filters are specified, uploads ALL stations with local data.
        Use data_sources, campaign_names, and station_names to filter stations.
        Filters work together to narrow down the selection (AND logic).
        Only stations with local raw data files will be uploaded.

    \b
    Upload Behavior:
        By default, upload is skipped if data already exists on a remote location.
        Use '--force True' to upload even if data already exists remotely.
        Warning: Forcing upload may create duplicate versions on the remote platform.

    \b
    Upload Options:
        --platform: Remote platform name (default: 'sandbox.zenodo' for testing)
        Use 'zenodo' for final data dissemination.
        --force: Upload even if data already exists remotely (default: False)

    \b
    Archive Directories:
        --data_archive_dir: Custom path to DISDRODB data archive
        --metadata_archive_dir: Custom path to DISDRODB metadata archive
        If not specified, paths from the active DISDRODB configuration are used

    \b
    Examples:

        # Upload a specific data sources to Zenodo Sandbox
        disdrodb_upload_archive --data_sources 'YOUR_DATA_SOURCE' --platform sandbox.zenodo

        # Upload a specific data sources to Zenodo and force upload
        disdrodb_upload_archive --data_sources 'YOUR_DATA_SOURCE' --platform zenodo --force True

        # Upload specific campaigns to Zenodo
        disdrodb_upload_archive --campaign_names 'HYMEX_LTE_SOP2 IFLOODS' --platform zenodo
    """  # noqa: D301
    from disdrodb.data_transfer.upload_data import upload_archive

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    upload_archive(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations options
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Upload options
        platform=platform,
        force=force,
    )
