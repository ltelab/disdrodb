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
    r"""Create summary figures and tables for a specific set of DISDRODB stations.

    Parameters \n
    ---------- \n
    data_sources : str
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        To specify multiple data sources, write i.e.: --data_sources 'GPM EPFL NCAR'
    campaign_names : str
        Name of the campaign(s) for which to create stations summaries.
        The name(s) must be UPPER CASE.
        To specify multiple campaigns, write i.e.: --campaign_names 'IPEX IMPACTS'
    station_names : str
        Station names.
        To specify multiple stations, write i.e.: --station_names 'station1 station2'
    data_archive_dir : str \n
        DISDRODB Data Archive directory \n
        Format: <...>/DISDRODB \n
        If not specified, uses path specified in the DISDRODB active configuration. \n
    """
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
