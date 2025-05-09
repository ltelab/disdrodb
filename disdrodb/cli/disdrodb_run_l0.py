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
"""Script to run the DISDRODB L0 processing."""
import sys
from typing import Optional

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
@click_processing_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
def disdrodb_run_l0(
    # Stations options
    data_sources: Optional[str] = None,
    campaign_names: Optional[str] = None,
    station_names: Optional[str] = None,
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
    """
    Run the L0 processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    data_archive_dir : str
        DISDRODB Data Archive directory
        Format: <...>/DISDRODB
    data_sources : str
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        To specify multiple data sources, write i.e.: --data_sources 'GPM EPFL NCAR'
    campaign_names : str
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        To specify multiple campaigns, write i.e.: --campaign_names 'IPEX IMPACTS'
    station_names : str
        Station names.
        To specify multiple stations, write i.e.: --station_names 'station1 station2'
    l0a_processing : bool
        Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
        The default is True.
    l0b_processing : bool
        Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
        The default is True.
    l0c_processing : bool
        Whether to launch processing to generate L0C netCDF4 file(s) from L0C data.
        The default is True.
    remove_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default is False.
    remove_l0b : bool
         Whether to remove the L0B files after having produced L0C netCDF files.
        The default is False.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is True.
    parallel : bool
        If True, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with os.cpu_count().
        However, you can customize it by typing i.e. DASK_NUM_WORKERS=4 disdrodb_run_l0
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files.
        For L0B, it processes just the first 100 rows of 3 L0A files.
        The default is False.
    data_archive_dir : str
        DISDRODB Data Archive directory
        Format: <...>/DISDRODB
        If not specified, uses path specified in the DISDRODB active configuration.
    """
    from disdrodb.routines import run_l0

    # Parse data_sources, campaign_names and station arguments
    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    # Run processing
    run_l0(
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
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
