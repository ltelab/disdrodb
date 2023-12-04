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
"""Routines to upload station data to the DISDRODB Decentralized Data Archive."""

import sys

import click

from disdrodb.data_transfer.upload_data import click_upload_options
from disdrodb.utils.scripts import click_base_dir_option, click_station_arguments, parse_base_dir

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_station_arguments
@click_upload_options
@click_base_dir_option
def disdrodb_upload_station(
    data_source: str,
    campaign_name: str,
    station_name: str,
    platform: str = None,
    base_dir: str = None,
    force: bool = False,
):
    from disdrodb.data_transfer.upload_data import upload_station

    base_dir = parse_base_dir(base_dir)
    upload_station(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        platform=platform,
        force=force,
    )
