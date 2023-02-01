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

sys.tracebacklimit = 0  # avoid full traceback error if occur
import click
from disdrodb.L0.L0_processing import (
    click_l0_processing_options,
    click_l0_station_arguments,
    click_l0_archive_options,
)

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator


@click.command()
@click_l0_station_arguments
@click_l0_processing_options
@click_l0_archive_options
def run_disdrodb_l0_station(
    # Station arguments
    disdrodb_dir,
    data_source,
    campaign_name,
    station_name,
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
    """Run the L0 processing of a specific DISDRODB station from the terminal.

    Parameters
    ----------
    disdrodb_dir : str
        Base directory of DISDRODB
        Format: <...>/DISDRODB
    data_source : str
        Institution name (when campaign data spans more than 1 country),
        or country (when all campaigns (or sensor networks) are inside a given country).
        Must be UPPER CASE.
    campaign_name : str
        Campaign name. Must be UPPER CASE.
    station_name : str
        Station name
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
         It takes places only if l0b_concat=True
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
        However, you can customize it by typing: DASK_NUM_WORKERS=4 run_disdrodb_l0_station
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files for each station.
        For L0B, it processes just the first 100 rows of 3 L0A files for each station.
        The default is False.
    """
    from disdrodb.L0.L0_processing import run_disdrodb_l0_station

    run_disdrodb_l0_station(
        disdrodb_dir=disdrodb_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
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
    run_disdrodb_l0_station()
