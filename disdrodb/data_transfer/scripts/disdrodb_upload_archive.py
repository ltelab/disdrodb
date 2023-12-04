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

import click

from disdrodb.data_transfer.upload_data import click_upload_archive_options, click_upload_options
from disdrodb.utils.scripts import click_base_dir_option, parse_arg_to_list, parse_base_dir

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_upload_archive_options
@click_upload_options
@click_base_dir_option
def disdrodb_upload_archive(
    base_dir: str = None,
    data_sources: str = None,
    campaign_names: str = None,
    station_names: str = None,
    platform: str = None,
    force: bool = False,
):
    from disdrodb.data_transfer.upload_data import upload_archive

    base_dir = parse_base_dir(base_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    upload_archive(
        base_dir=base_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        platform=platform,
        force=force,
    )
