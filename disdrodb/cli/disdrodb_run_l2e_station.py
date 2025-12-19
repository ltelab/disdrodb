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
"""Script to run the DISDRODB L2E station processing."""

import sys

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
def disdrodb_run_l2e_station(
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
    data_archive_dir: str | None = None,
    metadata_archive_dir: str | None = None,
):
    """Run the DISDRODB L2E processing chain for a specific DISDRODB station.

    It produces DISDRODB L2E files from existing DISDRODB L1 data of the specified station.
    DISDRODB L1 files must be available before launching the DISDRODB L2E processing.

    The DISDRODB L2E processing chain filters the disdrometer raw spectra,
    compute empirical integral DSD variables and optionally simulate
    polarimetric radar variables at multiple frequencies.

    \b
    Station Specification:
        Requires exact specification of data_source, campaign_name, and station_name.
        All three parameters must be provided and are case-sensitive (UPPER CASE required).

    \b
    Performance Options:
        --parallel: Uses multiple processes for faster processing (default: True)
        If parallel processing is enabled, each process will use a single thread
        to avoid issues with the HDF/netCDF library.
        The DASK_NUM_WORKERS environment variable controls the number of processes
        to use.A sensible default is automatically set by the software.
        --debugging_mode: Processes only a subset of data for testing
        --force: Overwrites existing output files (default: False)

    \b
    Examples:
        # Process a single station with full processing chain
        disdrodb_run_l2e_station EPFL HYMEX_LTE_SOP2 10

        # Force overwrite existing files with verbose output
        disdrodb_run_l2e_station EPFL HYMEX_LTE_SOP2 10 --force True --verbose True

        # Process station with debugging mode and custom workers
        DASK_NUM_WORKERS=4 disdrodb_run_l2e_station NETHERLANDS DELFT PAR001_Cabauw --debugging_mode True

    \b
    Important Notes:
        - Data source, campaign, and station names must be UPPER CASE
        - All three station identifiers are required (no wildcards or filtering)
    """  # noqa: D301
    from disdrodb.routines.l2 import run_l2e_station
    from disdrodb.utils.dask import close_dask_cluster, initialize_dask_cluster

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)

    # -------------------------------------------------------------------------.
    # If parallel=True, set the dask environment
    if parallel:
        cluster, client = initialize_dask_cluster(minimum_memory="4GB")

    # -------------------------------------------------------------------------.
    run_l2e_station(
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
