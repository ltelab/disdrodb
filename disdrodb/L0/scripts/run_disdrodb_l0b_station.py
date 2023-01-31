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

import click
from disdrodb.L0.L0_processing import click_l0_processing_options, click_l0_station_arguments
# -------------------------------------------------------------------------.
# Click Command Line Interface decorator


@click.command()
@click_l0_station_arguments
@click_l0_processing_options
def run_disdrodb_l0b_station(
    # Station arguments
    disdrodb_dir,
    data_source,
    campaign_name,
    station,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):
    """Run the L0B processing of a specific DISDRODB station from the terminal.

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
    station : str 
        Station name
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    parallel : bool
        If True, the files are processed simultanously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with os.cpu_count().
        However, you can customize it by typing: DASK_NUM_WORKERS=4 run_disdrodb_l0b_station
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    """
    from disdrodb.api.io import _get_disdrodb_directory
    from disdrodb.L0.L0_processing import run_l0b

    # Define processed dir
    processed_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level="L0B",
        data_source=data_source,
        campaign_name=campaign_name,
        check_exist=False,
    )
    run_l0b(
        processed_dir=processed_dir,
        station=station,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
    return None


if __name__ == "__main__":
    run_disdrodb_l0b_station()
