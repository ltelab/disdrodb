#!/usr/bin/env python3

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
"""Implement DISDRODB L1 processing."""

import datetime
import logging
import os
import time
from typing import Optional

import dask
import xarray as xr

# Directory
from disdrodb.api.create_directories import (
    create_logs_directory,
    create_product_directory,
)
from disdrodb.api.io import find_files
from disdrodb.api.path import (
    define_file_folder_path,
    define_l1_filename,
)
from disdrodb.api.search import get_required_product
from disdrodb.configs import get_data_archive_dir, get_folder_partitioning, get_metadata_archive_dir
from disdrodb.l1.processing import generate_l1
from disdrodb.utils.decorators import delayed_if_parallel, single_threaded_if_parallel

# Logger
from disdrodb.utils.logger import (
    close_logger,
    create_logger_file,
    create_product_logs,
    log_error,
    log_info,
)
from disdrodb.utils.writer import write_product

logger = logging.getLogger(__name__)


def get_l1_options():
    """Get L1 options."""
    # - TODO: from YAML
    # - TODO: as function of sensor name

    # minimum_diameter
    # --> PWS100: 0.05
    # --> PARSIVEL: 0.2495
    # --> RD80: 0.313
    # --> LPM: 0.125 (we currently discard first bin with this setting)

    # maximum_diameter
    # LPM: 8 mm
    # RD80: 5.6 mm
    # OTT: 26 mm

    l1_options = {
        # Fall velocity option
        "fall_velocity_method": "Beard1976",
        # Diameter-Velocity Filtering Options
        "minimum_diameter": 0.2495,  # OTT PARSIVEL first two bin no data !
        "maximum_diameter": 10,
        "minimum_velocity": 0,
        "maximum_velocity": 12,
        "above_velocity_fraction": 0.5,
        "above_velocity_tolerance": None,
        "below_velocity_fraction": 0.5,
        "below_velocity_tolerance": None,
        "small_diameter_threshold": 1,  # 2
        "small_velocity_threshold": 2.5,  # 3
        "maintain_smallest_drops": True,
    }
    return l1_options


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l1(
    filepath,
    data_dir,
    logs_dir,
    campaign_name,
    station_name,
    # Processing options
    force,
    verbose,
    parallel,  # this is used only to initialize the correct logger !
):
    """Generate the L1 product from the DISRODB L0C netCDF file.

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
    # -----------------------------------------------------------------.
    # Define product name
    product = "L1"

    # Define folder partitioning
    folder_partitioning = get_folder_partitioning()

    # -----------------------------------------------------------------.
    # Create file logger
    filename = os.path.basename(filepath)
    logger, logger_filepath = create_logger_file(
        logs_dir=logs_dir,
        filename=filename,
        parallel=parallel,
    )

    ##------------------------------------------------------------------------.
    # Log start processing
    msg = f"{product} processing of {filename} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    # Retrieve L1 configurations
    l1_options = get_l1_options()

    ##------------------------------------------------------------------------.
    ### Core computation
    try:
        # Open the raw netCDF
        with xr.open_dataset(filepath, chunks={}, decode_timedelta=False, cache=False) as ds:
            ds = ds[["raw_drop_number"]].load()

        # Produce L1 dataset
        ds = generate_l1(ds=ds, **l1_options)

        # Write L1 netCDF4 dataset
        if ds["time"].size > 1:
            # Define filepath
            filename = define_l1_filename(ds, campaign_name=campaign_name, station_name=station_name)
            folder_path = define_file_folder_path(ds, data_dir=data_dir, folder_partitioning=folder_partitioning)
            filepath = os.path.join(folder_path, filename)
            # Write to disk
            write_product(ds, product=product, filepath=filepath, force=force)

        ##--------------------------------------------------------------------.
        # Clean environment
        del ds

        # Log end processing
        msg = f"{product} processing of {filename} has ended."
        log_info(logger=logger, msg=msg, verbose=verbose)

    ##--------------------------------------------------------------------.
    # Otherwise log the error
    except Exception as e:
        error_type = str(type(e).__name__)
        msg = f"{error_type}: {e}"
        log_error(logger, msg, verbose=verbose)

    # Close the file logger
    close_logger(logger)

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
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
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

    # Define logs directory
    logs_dir = create_logs_directory(
        product=product,
        data_archive_dir=data_archive_dir,
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

    # ------------------------------------------------------------------------.
    # Create directory structure
    data_dir = create_product_directory(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        force=force,
    )

    # -------------------------------------------------------------------------.
    # List files to process
    required_product = get_required_product(product)
    flag_not_available_data = False
    try:
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=required_product,
            # Processing options
            debugging_mode=debugging_mode,
        )
    except Exception as e:
        print(str(e))  # Case where no file paths available
        flag_not_available_data = True

    # -------------------------------------------------------------------------.
    # If no data available, print error message and return None
    if flag_not_available_data:
        msg = (
            f"{product} processing of {data_source} {campaign_name} {station_name}"
            + f"has not been launched because of missing {required_product} data."
        )
        print(msg)
        return

    # -----------------------------------------------------------------.
    # Generate L1 files
    # - Loop over the L0 netCDF files and generate L1 files.
    # - If parallel=True, it does that in parallel using dask.delayed
    list_tasks = [
        _generate_l1(
            filepath=filepath,
            data_dir=data_dir,
            logs_dir=logs_dir,
            campaign_name=campaign_name,
            station_name=station_name,
            # Processing options
            force=force,
            verbose=verbose,
            parallel=parallel,
        )
        for filepath in filepaths
    ]
    list_logs = dask.compute(*list_tasks) if parallel else list_tasks

    # -----------------------------------------------------------------.
    # Define L1 summary logs
    create_product_logs(
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        data_archive_dir=data_archive_dir,
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
