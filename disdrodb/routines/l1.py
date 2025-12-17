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
"""Implement DISDRODB L1 processing."""

import datetime
import logging
import os
import time

import pandas as pd

from disdrodb.api.checks import check_station_inputs
from disdrodb.api.create_directories import (
    create_logs_directory,
    create_product_directory,
)
from disdrodb.api.io import open_netcdf_files
from disdrodb.api.path import (
    define_file_folder_path,
    define_l1_filename,
)
from disdrodb.api.search import get_required_product
from disdrodb.configs import (
    get_data_archive_dir,
    get_folder_partitioning,
    get_metadata_archive_dir,
)
from disdrodb.constants import METEOROLOGICAL_VARIABLES
from disdrodb.l1.classification import TEMPERATURE_VARIABLES
from disdrodb.l1.processing import generate_l1
from disdrodb.l1.resampling import resample_dataset
from disdrodb.metadata.reader import read_station_metadata
from disdrodb.routines.options import L1ProcessingOptions
from disdrodb.utils.dask import execute_tasks_safely
from disdrodb.utils.decorators import delayed_if_parallel, single_threaded_if_parallel

# Logger
from disdrodb.utils.logger import (
    create_product_logs,
    log_info,
)
from disdrodb.utils.routines import run_product_generation, try_get_required_filepaths
from disdrodb.utils.time import (
    ensure_sample_interval_in_seconds,
)
from disdrodb.utils.writer import write_product

logger = logging.getLogger(__name__)


def define_l1_logs_filename(campaign_name, station_name, start_time, end_time, temporal_resolution):
    """Define L1 logs filename."""
    starting_time = pd.to_datetime(start_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(end_time).strftime("%Y%m%d%H%M%S")
    logs_filename = f"L1.{temporal_resolution}.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}"
    return logs_filename


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l1(
    start_time,
    end_time,
    filepaths,
    data_dir,
    logs_dir,
    logs_filename,
    folder_partitioning,
    campaign_name,
    station_name,
    # L1 options
    temporal_resolution,
    product_options,
    # Processing options
    force,
    verbose,
    parallel,  # this is used only to initialize the correct logger !
):
    """Generate the L1 product from the DISDRODB L0C netCDF file.

    Parameters
    ----------
    filepath : str
        Path to the L0C netCDF file.
    data_dir : str
        Directory where the L1 netCDF file will be saved.
    logs_dir : str
        Directory where the log file will be saved.
    campaign_name : str
        Name of the campaign.
    station_name : str
        Name of the station.
    force : bool
        If True, overwrite existing files.
    verbose : bool
        Whether to verbose the processing.

    Returns
    -------
    str
        Path to the log file generated during processing.

    Notes
    -----
    If an error occurs during processing, it is caught and logged,
    but no error is raised to interrupt the execution.
    """
    # Define product
    product = "L1"
    # Define folder partitioning
    folder_partitioning = get_folder_partitioning()

    # Define product processing function
    def core(
        filepaths,
        start_time,
        end_time,
        campaign_name,
        station_name,
        # Processing options
        # logger,
        # verbose,
        force,
        # Product options
        temporal_resolution,
        product_options,
        # Archiving arguments
        data_dir,
        folder_partitioning,
    ):
        """Define L1 product processing."""
        # Define variables to load
        # - precip_flag used for OceanRain ODM470 data only
        # - Missing variables in dataset are simply not selected
        variables = ["raw_drop_number", "qc_time", "precip_flag", *TEMPERATURE_VARIABLES, *METEOROLOGICAL_VARIABLES]

        # Open the L0C netCDF files
        # with dask.config.set(scheduler="synchronous"):
        ds = open_netcdf_files(
            filepaths,
            start_time=start_time,
            end_time=end_time,
            variables=variables,
            parallel=False,
            compute=True,
        )

        # Define sample interval in seconds
        sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"]).to_numpy().item()

        # Resample dataset
        ds = resample_dataset(
            ds=ds,
            sample_interval=sample_interval,
            temporal_resolution=temporal_resolution,
        )

        # Produce L1 dataset
        ds = generate_l1(ds=ds, **product_options)

        # Ensure at least 1 timestep available
        if ds["time"].size <= 1:
            return None

        # Write L1 netCDF4 dataset
        filename = define_l1_filename(
            ds,
            campaign_name=campaign_name,
            station_name=station_name,
            temporal_resolution=temporal_resolution,
        )
        folder_path = define_file_folder_path(ds, dir_path=data_dir, folder_partitioning=folder_partitioning)
        filepath = os.path.join(folder_path, filename)
        write_product(ds, filepath=filepath, force=force)

        # Return L1 dataset
        return ds

    # Define product processing function kwargs
    core_func_kwargs = dict(  # noqa: C408
        filepaths=filepaths,
        start_time=start_time,
        end_time=end_time,
        campaign_name=campaign_name,
        station_name=station_name,
        # Processing options
        # verbose=verbose,
        force=force,
        # Product options
        temporal_resolution=temporal_resolution,
        product_options=product_options,
        # Archiving options
        data_dir=data_dir,
        folder_partitioning=folder_partitioning,
    )

    # TODO: Inspect core arguments: pass logger, verbose, folder_partitioning if present?
    # Run product generation
    logger_filepath = run_product_generation(
        product=product,
        logs_dir=logs_dir,
        logs_filename=logs_filename,
        parallel=parallel,
        verbose=verbose,
        folder_partitioning=folder_partitioning,
        core_func=core,
        core_func_kwargs=core_func_kwargs,
        pass_logger=False,
    )
    # Return the logger file path
    return logger_filepath


def run_l1_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = True,
    parallel: bool = True,
    debugging_mode: bool = False,
    # DISDRODB root directories
    data_archive_dir: str | None = None,
    metadata_archive_dir: str | None = None,
):
    """
    Run the L1 processing of a specific DISDRODB station when invoked from the terminal.

    The L1 routines just filter the raw drop spectrum and compute basic statistics.
    The L1 routine expects as input L0C files where each file has a unique sample interval.

    This function is intended to be called through the ``disdrodb_run_l1_station``
    command-line interface.

    Parameters
    ----------
    data_source : str
        The name of the institution (for campaigns spanning multiple countries) or
        the name of the country (for campaigns or sensor networks within a single country).
        Must be provided in UPPER CASE.
    campaign_name : str
        The name of the campaign. Must be provided in UPPER CASE.
    station_name : str
        The name of the station.
    force : bool, optional
        If ``True``, existing data in the destination directories will be overwritten.
        If ``False`` (default), an error will be raised if data already exists in the destination directories.
    verbose : bool, optional
        If ``True`` (default), detailed processing information will be printed to the terminal.
        If ``False``, less information will be displayed.
    parallel : bool, optional
        If ``True``, files will be processed in multiple processes simultaneously,
        with each process using a single thread to avoid issues with the HDF/netCDF library.
        If ``False`` (default), files will be processed sequentially in a single process,
        and multi-threading will be automatically exploited to speed up I/O tasks.
    debugging_mode : bool, optional
        If ``True``, the amount of data processed will be reduced.
        Only the first 3 files will be processed. The default value is ``False``.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    """
    # Define product
    product = "L1"

    # Define base directory
    data_archive_dir = get_data_archive_dir(data_archive_dir)

    # Retrieve DISDRODB Metadata Archive directory
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)

    # Check valid data_source, campaign_name, and station_name
    check_station_inputs(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # ------------------------------------------------------------------------.
    # Start processing
    if verbose:
        t_i = time.time()
        msg = f"{product} processing of station {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -------------------------------------------------------------------------.
    # List files to process
    # - If no data available, print error message and return None
    required_product = get_required_product(product)
    filepaths = try_get_required_filepaths(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=required_product,
        # Processing options
        debugging_mode=debugging_mode,
    )
    if filepaths is None:
        return

    # -------------------------------------------------------------------------.
    # Read station metadata and sensor name
    metadata = read_station_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    sensor_name = metadata["sensor_name"]

    # -------------------------------------------------------------------------.
    # Retrieve L1 processing options
    l1_processing_options = L1ProcessingOptions(
        sensor_name=sensor_name,
        filepaths=filepaths,
        parallel=parallel,
    )

    # -------------------------------------------------------------------------.
    # Generate products for each temporal resolution
    # temporal_resolution = "1MIN"
    # temporal_resolution = "10MIN"
    for temporal_resolution in l1_processing_options.temporal_resolutions:
        # Print progress message
        msg = f"Production of {product} {temporal_resolution} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

        # Retrieve event info
        files_partitions = l1_processing_options.group_files_by_temporal_partitions(temporal_resolution)

        # Retrieve folder partitioning (for files and logs)
        folder_partitioning = l1_processing_options.get_folder_partitioning(temporal_resolution)

        # Retrieve product options
        product_options = l1_processing_options.get_product_options(temporal_resolution)

        # ------------------------------------------------------------------.
        # Create product directory
        data_dir = create_product_directory(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
            force=force,
            # Option for L1
            temporal_resolution=temporal_resolution,
        )

        # Define logs directory
        logs_dir = create_logs_directory(
            product=product,
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Option for L1
            temporal_resolution=temporal_resolution,
        )

        # ------------------------------------------------------------------.
        # Generate files
        # - Loop over the L0C netCDF files and generate L1 files.
        # - If parallel=True, it does that in parallel using dask.delayed
        list_tasks = [
            _generate_l1(
                start_time=event_info["start_time"],
                end_time=event_info["end_time"],
                filepaths=event_info["filepaths"],
                data_dir=data_dir,
                logs_dir=logs_dir,
                logs_filename=define_l1_logs_filename(
                    campaign_name=campaign_name,
                    station_name=station_name,
                    start_time=event_info["start_time"],
                    end_time=event_info["end_time"],
                    temporal_resolution=temporal_resolution,
                ),
                folder_partitioning=folder_partitioning,
                campaign_name=campaign_name,
                station_name=station_name,
                # L1 product options
                temporal_resolution=temporal_resolution,
                product_options=product_options,
                # Processing options
                force=force,
                verbose=verbose,
                parallel=parallel,
            )
            for event_info in files_partitions
        ]
        list_logs = execute_tasks_safely(list_tasks=list_tasks, parallel=parallel, logs_dir=logs_dir)

        # -----------------------------------------------------------------.
        # Define product summary logs
        create_product_logs(
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=data_archive_dir,
            # Product options
            temporal_resolution=temporal_resolution,
            # Logs list
            list_logs=list_logs,
        )

    # ---------------------------------------------------------------------.
    # End L1 processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
        msg = f"{product} processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)


####-------------------------------------------------------------------------------------------------------------------.
