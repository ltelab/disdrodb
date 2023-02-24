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
from disdrodb.l0.l0_processing import (
    click_l0_processing_options,
    click_l0_station_arguments,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator


@click.command()
@click_l0_station_arguments
@click_l0_processing_options
def run_disdrodb_l0a_station(
    # Station arguments
    disdrodb_dir,
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    parallel: bool = True,
    debugging_mode: bool = False,
):
    """Run the L0A processing of a specific DISDRODB station from the terminal.

    Note: this function is used also for processing raw netCDF into L0B format.

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
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is True.
    parallel : bool
        If True, the files are processed simultanously in multiple processes.
        Each process will use a single thread.
        By default, the number of process is defined with os.cpu_count().
        However, you can customize it by typing: DASK_NUM_WORKERS=4 run_disdrodb_l0a_station
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        It processes just the first 3 raw data files.
        The default is False.
    """
    import os
    import dask
    from dask.distributed import Client, LocalCluster
    from disdrodb.api.io import _get_disdrodb_directory
    from disdrodb.l0.l0_reader import get_station_reader

    # -------------------------------------------------------------------------.
    # If parallel=True, set the dask environment
    if parallel:
        # Set HDF5_USE_FILE_LOCKING to avoid going stuck with HDF
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # Retrieve the number of process to run
        available_workers = os.cpu_count() - 2  # if not set, all CPUs
        num_workers = dask.config.get("num_workers", available_workers)
        # Create dask.distributed local cluster
        cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=1,
            processes=True,
            # memory_limit='8GB',
            # silence_logs=False,
        )
        Client(cluster)
    # -------------------------------------------------------------------------.
    # Get reader
    reader = get_station_reader(
        disdrodb_dir=disdrodb_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Define raw_dir and process_dir
    raw_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
    )
    processed_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level="L0A",
        data_source=data_source,
        campaign_name=campaign_name,
        check_exist=False,
    )

    # Run L0A processing
    # --> The reader call the run_l0a within the custom defined reader function
    # --> For the special case of raw netCDF data, it calls the run_l0b_from_nc function
    reader(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        station_name=station_name,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
    # -------------------------------------------------------------------------.
    # Close the cluster
    if parallel:
        cluster.close()
    return None


if __name__ == "__main__":
    run_disdrodb_l0a_station()
