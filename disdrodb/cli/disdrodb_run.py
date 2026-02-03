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
"""Script to launch DISDRODB products generation for the entire DISDRODB Archive."""

import sys

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_l0_archive_options,
    click_metadata_archive_dir_option,
    click_processing_options,
    click_stations_options,
    parse_archive_dir,
    parse_arg_to_list,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_stations_options
@click_l0_archive_options
@click.option(
    "-l1",
    "--l1_processing",
    type=bool,
    show_default=True,
    default=True,
    help="Run L1 processing",
)
@click.option(
    "-l2e",
    "--l2e_processing",
    type=bool,
    show_default=True,
    default=True,
    help="Run L2E processing",
)
@click.option(
    "-l2m",
    "--l2m_processing",
    type=bool,
    show_default=True,
    default=False,
    help="Run L2M processing.",
)
@click_processing_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_run(
    # Stations options
    data_sources: str | None = None,
    campaign_names: str | None = None,
    station_names: str | None = None,
    # L0 archive options
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    l0c_processing: bool = True,
    remove_l0a: bool = False,
    remove_l0b: bool = False,
    # Higher level processing options
    l1_processing: bool = True,
    l2e_processing: bool = True,
    l2m_processing: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
    # DISDRODB root directories
    data_archive_dir: str | None = None,
    metadata_archive_dir: str | None = None,
):
    """Run the complete processing of DISDRODB stations.

    This function allows to launch the complete processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    \b
    Processing Levels:
        L0A: Raw data converted to DISDRODB standardized Apache Parquet format
        L0B: L0A data converted to DISDRODB standardized netCDF4 format
        L0C: Apply time QC and consolidate L0B data into (by default daily) netCDF files.
        L1:  Temporally resample L0C data to standard intervals (1-60 minutes), apply QC algorithms
        and determine precipitation phase and hydrometeors types.
        L2E: Compute empirical integral DSD variables (e.g., rain rate, liquid water content)
        L2M: Fit parametric DSD models and compute model-based integral DSD variables

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
        # Process all stations with full processing chain
        disdrodb_run

        # Process specific data sources, skip L0A and L2M
        disdrodb_run --data_sources 'USA EPFL' --l0a_processing False --l2m_processing False

        # Process specific campaigns with debugging mode
        disdrodb_run --campaign_names 'DELFT IMPACTS' --debugging_mode True

        # Process specific stations with custom number of workers
        DASK_NUM_WORKERS=8 disdrodb_run --data_sources 'NASA' --station_names 'apu01 apu02'

        # Force overwrite existing files, verbose output
        disdrodb_run --data_sources 'EPFL' --force True --verbose True

    \b
    Data Management:
        --remove_l0a: Delete L0A files after L0B processing (saves disk space)
        --remove_l0b: Delete L0B files after L0C processing (saves disk space)
        Use with caution - removed files cannot be recovered without reprocessing

    \b
    Important Notes:
        - Data source names must be UPPER CASE
        - Campaign names must be UPPER CASE
        - To specify multiple values, use space-separated strings in quotes
        - Processing chain: L0A → L0B → L0C → L1 → L2E → L2M
        - You can skip early or late processing levels, but not intermediate ones
        - Processing validates chain consistency and will raise errors for gaps
        - Large datasets may require significant disk space and processing time
        - Use --debugging_mode for initial testing with reduced data volumes
    """  # noqa: D301
    from disdrodb.routines import run

    # Parse data_sources, campaign_names and station arguments
    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    # Run processing
    run(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Stations options
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # L0 archive options
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        l0c_processing=l0c_processing,
        remove_l0a=remove_l0a,
        remove_l0b=remove_l0b,
        # Higher level processing options
        l1_processing=l1_processing,
        l2e_processing=l2e_processing,
        l2m_processing=l2m_processing,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
