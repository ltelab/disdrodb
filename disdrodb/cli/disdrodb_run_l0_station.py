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
    r"""Run the L0 processing of a specific DISDRODB station from the terminal.

    Parameters \n
    ---------- \n
    data_source : str \n
        Institution name (when campaign data spans more than 1 country), or country (when all campaigns (or sensor
        networks) are inside a given country).\n
        Must be UPPER CASE.\n
    campaign_name : str \n
        Campaign name. Must be UPPER CASE.\n
    station_name : str \n
        Station name \n
    l0a_processing : bool
        Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
        The default is True.\n
    l0b_processing : bool \n
        Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.\n
        The default is True.\n
    l0c_processing : bool
        Whether to launch processing to generate L0C netCDF4 file(s) from L0C data.
        The default is True.
    remove_l0a : bool \n
        Whether to keep the L0A files after having generated the L0B netCDF products.\n
        The default is False.\n
    remove_l0b : bool
         Whether to remove the L0B files after having produced L0C netCDF files.
        The default is False.
    force : bool \n
        If True, overwrite existing data into destination directories.\n
        If False, raise an error if there are already data into destination directories.\n
        The default is False.\n
    verbose : bool \n
        Whether to print detailed processing information into terminal.\n
        The default is True.\n
    parallel : bool \n
        If True, the files are processed simultaneously in multiple processes.\n
        Each process will use a single thread to avoid issues with the HDF/netCDF library.\n
        By default, the number of process is defined with os.cpu_count().\n
        However, you can customize it by typing: DASK_NUM_WORKERS=4 disdrodb_run_l0_station\n
        If False, the files are processed sequentially in a single process.\n
        If False, multi-threading is automatically exploited to speed up I/0 tasks.\n
    debugging_mode : bool \n
        If True, it reduces the amount of data to process.\n
        For L0A, it processes just the first 3 raw data files for each station.\n
        For L0B, it processes just the first 100 rows of 3 L0A files for each station.\n
        The default is False.\n
    data_archive_dir : str \n
        DISDRODB Data Archive directory \n
        Format: <...>/DISDRODB \n
        If not specified, uses path specified in the DISDRODB active configuration. \n
    """
    from disdrodb.routines import run_l0_station

    data_archive_dir = parse_archive_dir(data_archive_dir)

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
