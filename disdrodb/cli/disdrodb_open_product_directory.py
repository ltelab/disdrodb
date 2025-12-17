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
"""Routine to open the DISDRODB Data Archive station product directory."""

import sys

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_station_arguments,
    parse_archive_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click.argument("product", metavar="<product>")
@click_station_arguments
@click_data_archive_dir_option
def disdrodb_open_product_directory(
    product: str,
    data_source: str,
    campaign_name: str,
    station_name: str,
    data_archive_dir: str | None = None,
):
    """Open the DISDRODB Data Archive station product directory in the system file explorer.

    Opens the data archive directory for a specific product level (RAW, L0A, L0B, etc.)
    of a station using the system's default file manager, allowing you to browse
    the data files.

    \b
    Station Specification:
        Requires exact specification of product, data_source, campaign_name, and station_name.
        All parameters must be provided and are case-sensitive (UPPER CASE required).

    \b
    Product Levels:
        Valid products: RAW, L0A, L0B, L0C, L1, L2E, L2M
        Specify the product level you want to browse

    \b
    Archive Directory:
        --data_archive_dir: Custom path to DISDRODB data archive
        If not specified, the path from the active DISDRODB configuration is used

    \b
    Examples:
        # Open Raw data directory for a station
        disdrodb_open_product_directory RAW EPFL HYMEX_LTE_SOP2 10

        # Open L0B product directory
        disdrodb_open_product_directory L0B NASA IFLOODS apu01

        # Open L2M directory with custom archive path
        disdrodb_open_product_directory L2M EPFL HYMEX_LTE_SOP2 10 --data_archive_dir /path/to/DISDRODB

    \b
    Important Notes:
        - Product name must be valid: RAW, L0A, L0B, L0C, L1, L2E, or L2M
        - Data source, campaign, and station names must be in UPPER CASE
    """  # noqa: D301
    from disdrodb.api.io import open_product_directory

    data_archive_dir = parse_archive_dir(data_archive_dir)

    open_product_directory(
        data_archive_dir=data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
