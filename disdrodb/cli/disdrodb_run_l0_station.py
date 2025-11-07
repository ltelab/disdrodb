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
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_run_l0_station(
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
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Run the DISDRODB L0 processing chain for a specific DISDRODB station.

    It produces the DISDRODB L0A, L0B and L0C product files out of
    the raw data of the specified station.

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
    Data Management:
        --remove_l0a: Delete L0A files after L0B processing (saves disk space)
        --remove_l0b: Delete L0B files after L0C processing (saves disk space)
        Use with caution - removed files cannot be recovered without reprocessing

    \b
    Examples:
        # Process a single station with full processing chain
        disdrodb_run_l0_station EPFL HYMEX_LTE_SOP2 10

        # Force overwrite existing files with verbose output
        disdrodb_run_l0_station EPFL HYMEX_LTE_SOP2 10 --force True --verbose True

        # Process station with debugging mode and custom workers
        DASK_NUM_WORKERS=4 disdrodb_run_l0_station NETHERLANDS DELFT PAR001_Cabauw --debugging_mode True

        # Skip L0A processing (start from existing L0B data)
        disdrodb_run_l0_station NASA IFLOODS apu01 --l0a_processing False

    \b
    Important Notes:
        - Data source, campaign, and station names must be UPPER CASE
        - All three station identifiers are required (no wildcards or filtering)
    """  # noqa: D301
    from disdrodb.routines import run_l0_station

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)

    run_l0_station(
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
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
