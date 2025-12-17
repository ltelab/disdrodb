#!/usr/bin/env python3
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
"""Script to run the DISDRODB L0A processing."""

import sys

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    click_processing_options,
    click_stations_options,
    parse_archive_dir,
    parse_arg_to_list,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_stations_options
@click_processing_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_run_l0a(
    # Stations options
    data_sources: str | None = None,
    campaign_names: str | None = None,
    station_names: str | None = None,
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
    # DISDRODB root directories
    data_archive_dir: str | None = None,
    metadata_archive_dir: str | None = None,
):
    """Run the DISDRODB L0A processing chain for many/all DISDRODB stations.

    It produces a DISDRODB L0A Apache Parquet file for each raw data file of
    the specified stations, unless the raw data are already in netCDF format.
    In this latter case, the raw data are directly converted to DISDRODB L0B files.

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
        disdrodb_run_l0a

        # Process a specific data source and force overwrite existing files
        disdrodb_run_l0a --data_sources EPFL --force True --verbose True

        # Process specific data sources
        disdrodb_run_l0a --data_sources 'USA EPFL'

        # Process specific campaigns in debugging mode
        disdrodb_run_l0a --campaign_names 'DELFT IMPACTS' --debugging_mode True

        # Process specific stations with custom number of workers
        DASK_NUM_WORKERS=8 disdrodb_run_l0a --data_sources NASA --station_names 'apu01 apu02'

    \b
    Important Notes:
        - Data source names must be UPPER CASE
        - Campaign names must be UPPER CASE
        - To specify multiple values, use space-separated strings in quotes
        - Use --debugging_mode for initial testing with reduced data volumes
    """  # noqa: D301
    from disdrodb.routines import run_l0a

    # Parse data_sources, campaign_names and station arguments
    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    # Run processing
    run_l0a(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations options
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
