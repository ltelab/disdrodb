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
"""Script to create summary figures and tables for a DISDRODB stationn."""

import sys
from typing import Optional

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    click_stations_options,
    parse_archive_dir,
    parse_arg_to_list,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator


@click.command()
@click_stations_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
@click.option("-p", "--parallel", type=bool, show_default=True, default=False, help="Read files in parallel")
@click.option(
    "-t",
    "--temporal_resolution",
    type=str,
    show_default=True,
    default="1MIN",
    help="Temporal resolution of the L2E product to be used for the summary.",
)
def disdrodb_create_summary(
    # Stations options
    data_sources: Optional[str] = None,
    campaign_names: Optional[str] = None,
    station_names: Optional[str] = None,
    # Processing options:
    parallel=False,
    temporal_resolution="1MIN",
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Create summary figures and tables for DISDRODB stations.

    Generates summary visualizations and statistics from DISDRODB L2E data products.
    The DISDRODB L2E files must be available.

    \b
    Station Selection:
        If no station filters are specified, creates summaries for ALL available stations.
        Use data_sources, campaign_names, and station_names to filter stations.
        Filters work together to narrow down the selection (AND logic).

    \b
    Processing Options:
        --parallel: Reads files in parallel for faster processing (default: False)
        --temporal_resolution: Temporal resolution of L2E product to use (default: 1MIN)
        Valid temporal resolutions depend on available L2E products.

    \b
    Archive Directories:
        --data_archive_dir: Custom path to DISDRODB data archive
        --metadata_archive_dir: Custom path to DISDRODB metadata archive
        If not specified, paths from the active DISDRODB configuration are used

    \b
    Examples:
        # Create summaries for all stations
        disdrodb_create_summary

        # Create summaries for specific data sources
        disdrodb_create_summary --data_sources 'EPFL NASA'

        # Create summaries for specific campaigns
        disdrodb_create_summary --campaign_names 'HYMEX_LTE_SOP2 IFLOODS'

        # Create summaries for specific stations with custom temporal resolution
        disdrodb_create_summary --station_names 'apu01 apu02' --temporal_resolution 5MIN

        # Create summaries with custom archive directory
        disdrodb_create_summary --data_sources EPFL --data_archive_dir /path/to/DISDRODB

    \b
    Important Notes:
        - Data source names must be in UPPER CASE
        - Campaign names must be in UPPER CASE
        - To specify multiple values, use space-separated strings in quotes
        - Requires L2E data products to be available for the selected stations
    """  # noqa: D301
    from disdrodb.routines import create_summary
    from disdrodb.utils.dask import close_dask_cluster, initialize_dask_cluster

    data_archive_dir = parse_archive_dir(data_archive_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    # -------------------------------------------------------------------------.
    # If parallel=True, set the dask environment
    if parallel:
        cluster, client = initialize_dask_cluster()

    # -------------------------------------------------------------------------.
    create_summary(
        # Station arguments
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Options
        parallel=parallel,
        temporal_resolution=temporal_resolution,
        # DISDRODB root directory
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
    )

    # -------------------------------------------------------------------------.
    # Close the cluster
    if parallel:
        close_dask_cluster(cluster, client)
