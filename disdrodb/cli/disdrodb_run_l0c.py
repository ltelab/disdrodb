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
"""Script to run the DISDRODB L0C processing."""
import sys
from typing import Optional

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    click_processing_options,
    click_remove_l0b_option,
    click_stations_options,
    parse_archive_dir,
    parse_arg_to_list,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_stations_options
@click_processing_options
@click_remove_l0b_option
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_run_l0c(
    # Stations options
    data_sources: Optional[str] = None,
    campaign_names: Optional[str] = None,
    station_names: Optional[str] = None,
    # L0C processing options
    remove_l0b: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the DISDRODB L0C processing chain for many/all DISDRODB stations.

    It produces DISDRODB L0C files from existing DISDRODB L0B data of the specified stations.
    DISDRODB L0B files must be available before launching the DISDRODB L0C processing.

    The DISDRODB L0C processing chain performs time-based quality control by
    removing duplicate timesteps and correcting trailing-second offsets.
    Then, it consolidate L0B data into by default daily netCDF files.

    \b
    Station Selection:
        If no station filters are specified, processes ALL available stations.
        Use data_sources, campaign_names, and station_names to filter stations.
        Filters work together to narrow down the selection (AND logic).

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
        # Process all stations
        disdrodb_run_l0c

        # Process a specific data source and force overwrite existing files
        disdrodb_run_l0c --data_sources EPFL --force True --verbose True

        # Process specific data sources
        disdrodb_run_l0c --data_sources 'USA EPFL'

        # Process specific campaigns in debugging mode
        disdrodb_run_l0c --campaign_names 'DELFT IMPACTS' --debugging_mode True

        # Process specific stations with custom number of workers
        DASK_NUM_WORKERS=8 disdrodb_run_l0c --data_sources NASA --station_names 'apu01 apu02'

    \b
    Important Notes:
        - Data source names must be UPPER CASE
        - Campaign names must be UPPER CASE
        - To specify multiple values, use space-separated strings in quotes
        - Use --debugging_mode for initial testing with reduced data volumes
    """  # noqa: D301
    from disdrodb.routines import run_l0c

    # Parse data_sources, campaign_names and station arguments
    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    # Run processing
    run_l0c(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations options
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # L0C processing options
        remove_l0b=remove_l0b,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
