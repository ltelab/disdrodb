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
"""Script to launch all DISDRODB products generation for a given station."""

import sys

import click

from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_l0_archive_options,
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
    default=True,
    help="Run L2M processing.",
)
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_run_station(
    # Station arguments
    data_source: str,
    campaign_name: str,
    station_name: str,
    # L0 archive options
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    l0c_processing: bool = True,
    remove_l0a: bool = False,
    remove_l0b: bool = False,
    # Higher level processing options
    l1_processing: bool = True,
    l2e_processing: bool = True,
    l2m_processing: bool = True,
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
    # DISDRODB root directories
    data_archive_dir: str | None = None,
    metadata_archive_dir: str | None = None,
):
    """Run the complete processing of a specific DISDRODB station.

    This function processes a single station through the complete DISDRODB processing
    chain, from raw data ingestion to final derived products. All processing levels
    are executed in sequence for the specified station.

    \b
    Processing Levels:
        L0A: Raw data converted to DISDRODB standardized Apache Parquet format
        L0B: L0A data converted to DISDRODB standardized netCDF4 format
        L0C: Apply time QC and consolidate L0B data into (by default daily) netCDF files
        L1:  Temporally resample L0C data to standard intervals (1-60 minutes), apply QC algorithms
                and determine precipitation phase and hydrometeors types
        L2E: Compute empirical integral DSD variables (e.g., rain rate, liquid water content)
        L2M: Fit parametric DSD models and compute model-based integral DSD variables

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
        disdrodb_run_station EPFL HYMEX_LTE_SOP2 10

        # Process station, skip L0A processing (start from existing L0B data)
        disdrodb_run_station NASA IFLOODS apu01 --l0a_processing False

        # Process station with debugging mode and custom workers
        DASK_NUM_WORKERS=4 disdrodb_run_station NETHERLANDS DELFT PAR001_Cabauw --debugging_mode True

        # Process station, skip final L2M level
        disdrodb_run_station FRANCE ENPC_CARNOT Carnot_Pars1 --l2m_processing False

        # Force overwrite existing files with verbose output
        disdrodb_run_station EPFL HYMEX_2012 10 --force True --verbose True

    \b
    Data Management:
        --remove_l0a: Delete L0A files after L0B processing (saves disk space)
        --remove_l0b: Delete L0B files after L0C processing (saves disk space)
        Use with caution - removed files cannot be recovered without reprocessing

    \b
    Important Notes:
        - Data source, campaign, and station names must be UPPER CASE
        - All three station identifiers are required (no wildcards or filtering)
        - Processing chain: L0A → L0B → L0C → L1 → L2E → L2M
        - You can skip early or late processing levels, but not intermediate ones
        - Processing validates chain consistency and will raise errors for gaps
        - Station-specific processing may require significant time for large datasets
        - Use --debugging_mode for initial testing with reduced data volumes
    """  # noqa: D301
    from disdrodb.routines import run_station

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)

    run_station(
        # DISDRODB root directories
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        # Station arguments
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
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
