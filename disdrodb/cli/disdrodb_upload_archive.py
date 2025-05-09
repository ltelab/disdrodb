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
    """Find all stations containing local data and upload them to a remote repository.

    Parameters
    ----------
    platform: str, optional
        Name of the remote platform.
        The default platform is ``"sandbox.zenodo"`` (for testing purposes).
        Switch to ``"zenodo"`` for final data dissemination.
    force: bool, optional
        If ``True``, upload even if data already exists on another remote location.
        The default value is ``force=False``.
    data_archive_dir : str (optional)
        The directory path where the DISDRODB Data Archive is located.
        The directory path must end with ``<...>/DISDRODB``.
        If ``None``, it uses the ``data_archive_dir`` path specified
        in the DISDRODB active configuration.

    Other Parameters
    ----------------
    data_sources: str or list of str, optional
        Data source name (eg: EPFL).
        If not provided (``None``), all data sources will be uploaded.
        The default value is ``data_source=None``.
    campaign_names: str or list of str, optional
        Campaign name (eg:  EPFL_ROOF_2012).
        If not provided (``None``), all campaigns will be uploaded.
        The default value is ``campaign_name=None``.
    station_names: str or list of str, optional
        Station name.
        If not provided (``None``), all stations will be uploaded.
        The default value is ``station_name=None``.
    """
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
