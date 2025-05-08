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
"""Script to run the DISDRODB L2M station processing."""
import sys
from typing import Optional

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    click_processing_options,
    click_station_arguments,
    parse_archive_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator


@click.command()
@click_station_arguments
@click_processing_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_run_l2m_station(
    # Station arguments
    data_source: str,
    campaign_name: str,
    station_name: str,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    parallel: bool = True,
    debugging_mode: bool = False,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """
    Run the L2M processing of a specific DISDRODB station from the terminal.

    Parameters
    ----------
    data_source : str
        Institution name (when campaign data spans more than 1 country),
        or country (when all campaigns (or sensor networks) are inside a given country).
        Must be UPPER CASE.
    campaign_name : str
        Campaign name. Must be UPPER CASE.
    station_name : str
        Station name
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is True.
    parallel : bool
        If True, the files are processed simultaneously in multiple processes.
        Each process will use a single thread.
        By default, the number of process is defined with os.cpu_count().
        However, you can customize it by typing: DASK_NUM_WORKERS=4 disdrodb_run_l0a_station
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        It processes just the first 3 raw data files.
        The default is False.
    data_archive_dir : str
        DISDRODB Data Archive directory.
        Format: <...>/DISDRODB
        If not specified, uses path specified in the DISDRODB active configuration.
    """
    from disdrodb.l2.routines import run_l2m_station
    from disdrodb.utils.dask import close_dask_cluster, initialize_dask_cluster

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)

    # -------------------------------------------------------------------------.
    # If parallel=True, set the dask environment
    if parallel:
        cluster, client = initialize_dask_cluster()

    # -------------------------------------------------------------------------.
    run_l2m_station(
        # Station arguments
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
    )

    # -------------------------------------------------------------------------.
    # Close the cluster
    if parallel:
        close_dask_cluster(cluster, client)
