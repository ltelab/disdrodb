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
"""Script to run the DISDRODB L0 station processing."""
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
def disdrodb_create_summary_station(
    # Station arguments
    data_source: str,
    campaign_name: str,
    station_name: str,
    # Processing options:
    parallel=False,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
):
    r"""Create summary figures and tables for a specific DISDRODB station.

    Parameters \n
    ---------- \n
    data_source : str \n
        Institution name (when campaign data spans more than 1 country),
        or country (when all campaigns (or sensor networks) are inside a given country).\n
        Must be UPPER CASE.\n
    campaign_name : str \n
        Campaign name. Must be UPPER CASE.\n
    station_name : str \n
        Station name \n
    data_archive_dir : str \n
        DISDRODB Data Archive directory \n
        Format: <...>/DISDRODB \n
        If not specified, uses path specified in the DISDRODB active configuration. \n
    """
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
        # DISDRODB root directory
        data_archive_dir=data_archive_dir,
    )

    # -------------------------------------------------------------------------.
    # Close the cluster
    if parallel:
        close_dask_cluster(cluster, client)
