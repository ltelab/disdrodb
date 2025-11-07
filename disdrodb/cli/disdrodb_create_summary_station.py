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
"""Script to create summary figures and tables for a DISDRODB station."""
import sys
from typing import Optional

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_station_arguments,
    parse_archive_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator


@click.command()
@click_station_arguments
@click_data_archive_dir_option
@click.option("-p", "--parallel", type=bool, show_default=True, default=False, help="Read files in parallel")
@click.option(
    "-t",
    "--temporal_resolution",
    type=str,
    show_default=True,
    default="1MIN",
    help="Temporal resolution of the L2E product to be used for the summary.",
)
def disdrodb_create_summary_station(
    # Station arguments
    data_source: str,
    campaign_name: str,
    station_name: str,
    # Processing options:
    parallel=False,
    temporal_resolution="1MIN",
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
):
    """Create summary figures and tables for a specific DISDRODB station.

    Generates summary visualizations and statistics using the DISDRODB L2E product
    of the specified station.

    \b
    Station Specification:
        Requires exact specification of data_source, campaign_name, and station_name.
        All three parameters must be provided and are case-sensitive (UPPER CASE required).

    \b
    Processing Options:
        --parallel: Reads files in parallel for faster processing (default: False)
        --temporal_resolution: Temporal resolution of L2E product to use (default: 1MIN)
        Valid temporal resolutions depend on available L2E products

    \b
    Archive Directory:
        --data_archive_dir: Custom path to DISDRODB data archive
        If not specified, the path from the active DISDRODB configuration is used

    \b
    Examples:
        # Create summary for a specific station
        disdrodb_create_summary_station EPFL HYMEX_LTE_SOP2 10

        # Create summary with custom temporal resolution
        disdrodb_create_summary_station EPFL HYMEX_LTE_SOP2 10 --temporal_resolution 5MIN

        # Create summary with custom archive directory
        disdrodb_create_summary_station NASA IFLOODS apu01 --data_archive_dir /path/to/DISDRODB

    \b
    Important Notes:
        - Data source, campaign, and station names must be in UPPER CASE
        - All three station identifiers are required (no wildcards)
        - Requires L2E data products to be available for the specified station
    """  # noqa: D301
    from disdrodb.summary.routines import create_station_summary
    from disdrodb.utils.dask import close_dask_cluster, initialize_dask_cluster

    data_archive_dir = parse_archive_dir(data_archive_dir)

    # -------------------------------------------------------------------------.
    # If parallel=True, set the dask environment
    if parallel:
        cluster, client = initialize_dask_cluster()

    # -------------------------------------------------------------------------.
    create_station_summary(
        # Station arguments
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Options
        parallel=parallel,
        temporal_resolution=temporal_resolution,
        # DISDRODB root directory
        data_archive_dir=data_archive_dir,
    )

    # -------------------------------------------------------------------------.
    # Close the cluster
    if parallel:
        close_dask_cluster(cluster, client)
