# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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
import sys
import click
from disdrodb.utils.scripts import parse_arg_to_list
from disdrodb.l0.l0_processing import (
    click_l0_processing_options,
    click_l0_stations_options,
    click_l0_archive_options,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click.argument("disdrodb_dir", metavar="<disdrodb_dir>")
@click_l0_stations_options
@click_l0_archive_options
@click_l0_processing_options
def run_disdrodb_l0(
    disdrodb_dir,
    # L0 disdrodb stations options
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # L0 archive options
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    l0b_concat: bool = True,
    remove_l0a: bool = False,
    remove_l0b: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
):
    """Run the L0 processing of DISDRODB stations.

    This function enable to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    disdrodb_dir : str
        Base directory of DISDRODB
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
    l0b_concat : bool
        Whether to concatenate all raw files into a single L0B netCDF file.
        If l0b_concat=True, all raw files will be saved into a single L0B netCDF file.
        If l0b_concat=False, each raw file will be converted into the corresponding L0B netCDF file.
        The default is False.
    remove_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default is False.
    remove_l0b : bool
         Whether to remove the L0B files after having concatenated all L0B netCDF files.
         It takes places only if l0b_concat = True
        The default is False.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is True.
    parallel : bool
        If True, the files are processed simultanously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with os.cpu_count().
        However, you can customize it by typing i.e. DASK_NUM_WORKERS=4 run_disdrodb_l0
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files.
        For L0B, it processes just the first 100 rows of 3 L0A files.
        The default is False.
    """
    from disdrodb.l0.l0_processing import run_disdrodb_l0

    # Parse data_sources, campaign_names and station arguments
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    # Run processing
    run_disdrodb_l0(
        disdrodb_dir=disdrodb_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # L0 archive options
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        l0b_concat=l0b_concat,
        remove_l0a=remove_l0a,
        remove_l0b=remove_l0b,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
    return None


if __name__ == "__main__":
    run_disdrodb_l0()
