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
"""Script to run the DISDRODB L0A processing."""
import sys
from typing import Optional

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
    data_sources: Optional[str] = None,
    campaign_names: Optional[str] = None,
    station_names: Optional[str] = None,
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the L0A processing of DISDRODB stations.

    This function launches the L0A processing for many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the L0A conversion
    for the stations matching the provided ``data_sources``, ``campaign_names`` and
    ``station_names`` filters.

    \b
    Processing Level - L0A:
        L0A converts raw instrument files to the DISDRODB standardized Apache Parquet format.
        This is the first step in the processing chain (L0A → L0B → L0C → L1 → L2E → L2M).

    \b
    Station Selection:
        If no station filters are specified, processes ALL available stations.
        Use ``data_sources``, ``campaign_names``, and ``station_names`` to filter stations.
        Filters work together to narrow down the selection (AND logic).

    \b
    Performance Options:
        --parallel: Uses multiple processes for faster processing (default: True).
        If --parallel is enabled, each process will use a single thread to avoid issues
        with the HDF/netCDF libraries (where applicable).

        --debugging_mode: Processes only a small subset of data for quick testing (default: False).
        --force: Overwrites existing output files (default: False).

        The ``DASK_NUM_WORKERS`` environment variable controls the number of worker
        processes used when ``--parallel`` is enabled. A sensible default is set
        automatically when a cluster is initialized.

    \b
    Examples:
        # Process all stations L0A conversion
        disdrodb_run_l0a

        # Process specific data sources with debugging mode
        disdrodb_run_l0a --data_sources 'NASA EPFL' --debugging_mode

        # Process specific stations with custom number of workers
        DASK_NUM_WORKERS=8 disdrodb_run_l0a --data_sources 'NASA' --station_names 'apu01'

        # Force overwrite existing files, verbose output
        disdrodb_run_l0a --data_sources 'EPFL' --force --verbose

    \b
    Data Management:
        L0A is an input step for downstream processing. Deleting L0A files should
        only be done if you can re-run conversion from raw data when needed.

    \b
    Important Notes:
        - Data source names must be UPPER CASE.
        - Campaign names must be UPPER CASE.
        - To specify multiple values, use space-separated strings in quotes.
        - Use ``--debugging_mode`` for initial testing with reduced data volumes.

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
