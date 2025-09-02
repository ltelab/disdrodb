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
"""Implement DISDRODB L0 processing."""

import datetime
import logging
import os
import time
from typing import Optional

import dask

from disdrodb.api.checks import check_sensor_name, check_station_inputs
from disdrodb.api.create_directories import (
    create_l0_directory_structure,
    create_logs_directory,
    create_product_directory,
)
from disdrodb.api.io import find_files, remove_product
from disdrodb.api.path import (
    define_file_folder_path,
    define_l0a_filename,
    define_l0b_filename,
    define_l0c_filename,
    define_metadata_filepath,
)
from disdrodb.api.search import get_required_product
from disdrodb.configs import get_data_archive_dir, get_folder_partitioning, get_metadata_archive_dir
from disdrodb.issue import read_station_issue
from disdrodb.l0.l0_reader import get_reader
from disdrodb.l0.l0a_processing import (
    read_l0a_dataframe,
    sanitize_df,
    write_l0a,
)
from disdrodb.l0.l0b_nc_processing import sanitize_ds
from disdrodb.l0.l0b_processing import (
    generate_l0b,
    set_l0b_encodings,
    write_l0b,
)
from disdrodb.l0.l0c_processing import (
    create_daily_file,
    get_files_per_days,
    retrieve_possible_measurement_intervals,
)
from disdrodb.metadata import read_station_metadata
from disdrodb.utils.attrs import set_disdrodb_attrs
from disdrodb.utils.decorators import delayed_if_parallel, single_threaded_if_parallel

# Logger
from disdrodb.utils.logger import (
    create_product_logs,
    log_info,
)
from disdrodb.utils.routines import run_product_generation
from disdrodb.utils.writer import write_product
from disdrodb.utils.yaml import read_yaml

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------.
#### Creation of L0A and L0B Single Station File


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l0a(
    filepath,
    data_dir,
    logs_dir,
    campaign_name,
    station_name,
    # Processing info
    reader,
    metadata,
    issue_dict,
    # Processing options
    force,
    verbose,
    parallel,
):
    """Generate L0A file from raw txt file."""
    # Define product
    product = "L0A"
    # Define folder partitioning
    folder_partitioning = get_folder_partitioning()
    # Retrieve sensor name
    sensor_name = metadata["sensor_name"]
    # Define logger filename
    logs_filename = os.path.basename(filepath)

    # Define product processing function
    def core(
        filepath,
        campaign_name,
        station_name,
        sensor_name,
        verbose,
        issue_dict,
        logger,
        data_dir,
        folder_partitioning,
    ):
        """Define L0A product processing."""
        # Read raw data into L0A format
        df = reader(filepath, logger=logger)
        df = sanitize_df(df, sensor_name=sensor_name, verbose=verbose, issue_dict=issue_dict, logger=logger)

        # Write L0A dataframe
        filename = define_l0a_filename(df, campaign_name=campaign_name, station_name=station_name)
        folder_path = define_file_folder_path(df, dir_path=data_dir, folder_partitioning=folder_partitioning)
        out_path = os.path.join(folder_path, filename)
        write_l0a(df, filepath=out_path, force=force, logger=logger, verbose=verbose)
        # Return L0A dataframe
        return df

    # Define product processing function kwargs
    core_func_kwargs = dict(  # noqa: C408
        filepath=filepath,
        campaign_name=campaign_name,
        station_name=station_name,
        data_dir=data_dir,
        folder_partitioning=folder_partitioning,
        sensor_name=sensor_name,
        verbose=verbose,
        issue_dict=issue_dict,
    )
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
        pass_logger=True,
    )
    return logger_filepath


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l0b_from_nc(
    filepath,
    data_dir,
    logs_dir,
    campaign_name,
    station_name,
    # Processing info
    reader,
    metadata,
    issue_dict,
    # Processing options
    force,
    verbose,
    parallel,
):
    """Generate L0B file from raw netCDF file."""
    # Define product
    product = "L0B"
    # Define folder partitioning
    folder_partitioning = get_folder_partitioning()
    # Define logger filename
    logs_filename = os.path.basename(filepath)

    # Define product processing function
    def core(
        filepath,
        campaign_name,
        station_name,
        metadata,
        issue_dict,
        logger,
        verbose,
        data_dir,
        folder_partitioning,
    ):
        """Define L0B product processing."""
        # Retrieve sensor name
        sensor_name = metadata["sensor_name"]
        # Read raw netCDF and sanitize to L0B format
        ds = reader(filepath, logger=logger)
        ds = sanitize_ds(
            ds=ds,
            sensor_name=sensor_name,
            metadata=metadata,
            issue_dict=issue_dict,
            verbose=verbose,
            logger=logger,
        )

        # Write L0B netCDF4 dataset
        filename = define_l0b_filename(ds=ds, campaign_name=campaign_name, station_name=station_name)
        folder_path = define_file_folder_path(ds, dir_path=data_dir, folder_partitioning=folder_partitioning)
        filepath = os.path.join(folder_path, filename)
        write_l0b(ds, filepath=filepath, force=force)

        # Return L0B dataset
        return ds

    # Define product processing function kwargs
    core_func_kwargs = dict(  # noqa: C408
        filepath=filepath,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata=metadata,
        issue_dict=issue_dict,
        verbose=verbose,
        data_dir=data_dir,
        folder_partitioning=folder_partitioning,
    )
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
        pass_logger=True,
    )
    return logger_filepath


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l0b(
    filepath,
    data_dir,
    logs_dir,
    campaign_name,
    station_name,
    # Processing info
    metadata,
    # Processing options
    force,
    verbose,
    parallel,
    debugging_mode,
):
    # Define product
    product = "L0B"
    # Define folder partitioning
    folder_partitioning = get_folder_partitioning()
    # Define logger filename
    logs_filename = os.path.basename(filepath)

    # Define product processing function
    def core(
        filepath,
        campaign_name,
        station_name,
        metadata,
        logger,
        debugging_mode,
        verbose,
        data_dir,
        folder_partitioning,
    ):
        """Define L0B product processing."""
        # Read L0A Apache Parquet file
        df = read_l0a_dataframe(filepath, debugging_mode=debugging_mode)
        # Create L0B xarray Dataset
        ds = generate_l0b(df=df, metadata=metadata, logger=logger, verbose=verbose)

        # Write L0B netCDF4 dataset
        filename = define_l0b_filename(ds=ds, campaign_name=campaign_name, station_name=station_name)
        folder_path = define_file_folder_path(ds, dir_path=data_dir, folder_partitioning=folder_partitioning)
        filepath = os.path.join(folder_path, filename)
        write_l0b(ds, filepath=filepath, force=force)
        # Return L0B dataset
        return ds

    # Define product processing function kwargs
    core_func_kwargs = dict(  # noqa: C408
        filepath=filepath,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata=metadata,
        debugging_mode=debugging_mode,
        verbose=verbose,
        data_dir=data_dir,
        folder_partitioning=folder_partitioning,
    )
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
        pass_logger=True,
    )
    # Return the logger file path
    return logger_filepath


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l0c(
    day,
    filepaths,
    data_dir,
    logs_dir,
    metadata_filepath,
    campaign_name,
    station_name,
    # Processing options
    force,
    verbose,
    parallel,  # this is used only to initialize the correct logger !
):
    """Define L0C product processing."""
    # Define product
    product = "L0C"
    # Define folder partitioning
    folder_partitioning = get_folder_partitioning()
    # Define logger filename
    logs_filename = day

    # Define product processing function
    def core(
        day,
        filepaths,
        metadata_filepath,
        campaign_name,
        station_name,
        logger,
        data_dir,
        folder_partitioning,
        verbose,
    ):
        """Define L0C product processing."""
        # Retrieve measurement_intervals
        # TODO: in future available from dataset (in attributes removed from L0C !)
        metadata = read_yaml(metadata_filepath)
        measurement_intervals = retrieve_possible_measurement_intervals(metadata)

        # Produce L0C datasets
        dict_ds = create_daily_file(
            day=day,
            filepaths=filepaths,
            measurement_intervals=measurement_intervals,
            ensure_variables_equality=True,
            logger=logger,
            verbose=verbose,
        )

        # Write a dataset for each sample interval
        for ds in dict_ds.values():  # (sample_interval, ds)
            # Write L0C netCDF4 dataset
            if ds["time"].size > 1:
                # Get sensor name from dataset
                sensor_name = ds.attrs.get("sensor_name")
                campaign_name = ds.attrs.get("campaign_name")
                station_name = ds.attrs.get("station_name")

                # Set encodings
                ds = set_l0b_encodings(ds=ds, sensor_name=sensor_name)
                # Update global attributes
                ds = set_disdrodb_attrs(ds, product=product)

                # Write L0C netCDF4 dataset
                filename = define_l0c_filename(ds, campaign_name=campaign_name, station_name=station_name)
                folder_path = define_file_folder_path(ds, dir_path=data_dir, folder_partitioning=folder_partitioning)
                filepath = os.path.join(folder_path, filename)
                write_product(ds, filepath=filepath, force=force)

        # Return L0C dataset
        return ds

    # Define product processing function kwargs
    core_func_kwargs = dict(  # noqa: C408
        filepaths=filepaths,
        day=day,
        metadata_filepath=metadata_filepath,
        campaign_name=campaign_name,
        station_name=station_name,
        data_dir=data_dir,
        folder_partitioning=folder_partitioning,
        verbose=verbose,
    )

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
        pass_logger=True,
    )
    # Return the logger file path
    return logger_filepath


####--------------------------------------------------------------------------.
#### DISDRODB Station Functions


def run_l0a_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    # DISDRODB root directories
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """
    Run the L0A processing of a specific DISDRODB station when invoked from the terminal.

    This function is intended to be called through the ``disdrodb_run_l0a_station``
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
        If ``True``, files will be processed in multiple processes simultaneously
        with each process using a single thread.
        If ``False`` (default), files will be processed sequentially in a single process,
        and multi-threading will be automatically exploited to speed up I/O tasks.
    debugging_mode : bool, optional
        If ``True``, the amount of data processed will be reduced.
        Only the first 3 raw data files will be processed. The default value is ``False``.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    """
    # Retrieve DISDRODB Metadata Archive and Data Archive root directories
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)

    # Check valid data_source, campaign_name, and station_name
    check_station_inputs(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # ------------------------------------------------------------------------.
    # Read metadata
    metadata = read_station_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # ------------------------------------------------------------------------.
    # Define raw data ingestion chain
    # --> If raw data are netCDF files, this routine produces directly L0B files
    # --> Otherwise, it produces L0A files.
    if metadata["raw_data_format"] == "netcdf":
        generate_standardized_files = _generate_l0b_from_nc
        product = "L0B"
    else:
        generate_standardized_files = _generate_l0a
        product = "L0A"

    # ------------------------------------------------------------------------.
    # Start product processing
    t_i = time.time()
    msg = f"{product} processing of station {station_name} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------------.
    # Create directory structure
    data_dir = create_l0_directory_structure(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,  # L0A or L0B
        force=force,
    )

    # -------------------------------------------------------------------------.
    # Define logs directory
    logs_dir = create_logs_directory(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # -----------------------------------------------------------------.
    # Read issue YAML file
    issue_dict = read_station_issue(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    ##------------------------------------------------------------------------.
    # Retrieve sensor name
    sensor_name = metadata["sensor_name"]
    check_sensor_name(sensor_name)

    # Retrieve sensor name
    reader_reference = metadata["reader"]

    # Retrieve glob patterns
    glob_pattern = metadata["raw_data_glob_pattern"]

    ##------------------------------------------------------------------------.
    # Retrieve reader
    reader = get_reader(reader_reference, sensor_name=sensor_name)

    # -------------------------------------------------------------------------.
    # List files to process
    filepaths = find_files(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
        debugging_mode=debugging_mode,
        data_archive_dir=data_archive_dir,
        glob_pattern=glob_pattern,
    )

    # Print the number of files to be processed
    n_files = len(filepaths)
    msg = f"{n_files} raw files are ready to be processed."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # -----------------------------------------------------------------.
    # Generate L0A/L0B files
    # - Loop over the files and save the L0A Apache Parquet files.
    # - If parallel=True, it does that in parallel using dask.delayed
    list_tasks = [
        generate_standardized_files(
            filepath=filepath,
            data_dir=data_dir,
            logs_dir=logs_dir,
            campaign_name=campaign_name,
            station_name=station_name,
            # Reader argument
            reader=reader,
            # Processing info
            metadata=metadata,
            issue_dict=issue_dict,
            # Processing options
            force=force,
            verbose=verbose,
            parallel=parallel,
        )
        for filepath in filepaths
    ]
    list_logs = dask.compute(*list_tasks) if parallel else list_tasks

    # -----------------------------------------------------------------.
    # Define product summary logs
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
    # End product processing
    timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
    msg = f"{product} processing of station {station_name} completed in {timedelta_str}"
    log_info(logger=logger, msg=msg, verbose=verbose)
    # ---------------------------------------------------------------------.


def run_l0b_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # L0B processing options
    remove_l0a: bool = False,
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
    Run the L0B processing of a specific DISDRODB station when invoked from the terminal.

    This function is intended to be called through the ``disdrodb_run_l0b_station``
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
        Only 100 rows sampled from 3 L0A files will be processed. The default value is ``False``.
    remove_l0a: bool, optional
        Whether to remove the processed L0A files. The default value is ``False``.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    """
    # Define product
    product = "L0B"

    # Retrieve DISDRODB base directory
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
    # -----------------------------------------------------------------.
    # Retrieve metadata
    metadata = read_station_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Skip run_l0b processing if the raw data are netCDFs
    # - L0B produced when running L0A ...
    if metadata["raw_data_format"] == "netcdf":
        return

    # -----------------------------------------------------------------.
    # Start L0B processing
    t_i = time.time()
    msg = f"{product} processing of station_name {station_name} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # -----------------------------------------------------------------.
    # Define logs directory
    logs_dir = create_logs_directory(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create product directory
    data_dir = create_product_directory(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        force=force,
    )

    ##----------------------------------------------------------------.
    # Get L0A files for the station
    required_product = get_required_product(product)
    flag_not_available_data = False
    try:
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=required_product,
            debugging_mode=debugging_mode,
        )
    except Exception as e:
        print(str(e))  # Case where no file paths available
        flag_not_available_data = True

    # -------------------------------------------------------------------------.
    # If no data available, print error message and return None
    if flag_not_available_data:
        msg = (
            f"{product} processing of {data_source} {campaign_name} {station_name} "
            + f"has not been launched because of missing {required_product} data."
        )
        print(msg)
        return

    ##----------------------------------------------------------------.
    # Generate L0B files
    # Loop over the L0A files and save the L0B netCDF files.
    # - If parallel=True, it does that in parallel using dask.bag
    #   Settings npartitions=len(filepaths) enable to wait prior task on a core
    #   finish before starting a new one.
    # BUG: If debugging_mode=True and parallel=True a subtle bug can currently occur when
    #   two processes with a subsetted L0A files want to create the same L0B files !
    list_tasks = [
        _generate_l0b(
            filepath=filepath,
            data_dir=data_dir,
            logs_dir=logs_dir,
            metadata=metadata,
            campaign_name=campaign_name,
            station_name=station_name,
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        for filepath in filepaths
    ]
    list_logs = dask.compute(*list_tasks) if parallel else list_tasks
    # if not parallel:
    #     list_logs = [
    #         _generate_l0b(
    #             filepath=filepath,
    #             data_dir=data_dir,
    #             logs_dir=logs_dir,
    #             metadata=metadata,
    #             campaign_name=campaign_name,
    #             station_name=station_name,
    #             force=force,
    #             verbose=verbose,
    #             debugging_mode=debugging_mode,
    #             parallel=parallel,
    #         )
    #         for filepath in filepaths
    #     ]

    # else:
    #     bag = db.from_sequence(filepaths, npartitions=len(filepaths))
    #     list_logs = bag.map(
    #         _generate_l0b,
    #         data_dir=data_dir,
    #         logs_dir=logs_dir,
    #         metadata=metadata,
    #         campaign_name=campaign_name,
    #         station_name=station_name,
    #         force=force,
    #         verbose=verbose,
    #         debugging_mode=debugging_mode,
    #         parallel=parallel,
    #     ).compute()

    # -----------------------------------------------------------------.
    # Define L0B summary logs
    create_product_logs(
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        data_archive_dir=data_archive_dir,
        # Logs list
        list_logs=list_logs,
    )

    # -----------------------------------------------------------------.
    # End L0B processing
    timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
    msg = f"{product} processing of station_name {station_name} completed in {timedelta_str}"
    log_info(logger=logger, msg=msg, verbose=verbose)

    # -----------------------------------------------------------------.
    # Option to remove L0A
    if remove_l0a:
        remove_product(
            data_archive_dir=data_archive_dir,
            product="L0A",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            logger=logger,
            verbose=verbose,
        )


def run_l0c_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # L0C processing options
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
    Run the L0C processing of a specific DISDRODB station when invoked from the terminal.

    The DISDRODB L0A and L0B routines just convert source raw data into netCDF format.
    The DISDRODB L0C routine ingests L0B files and performs data homogenization.
    The DISDRODB L0C routine takes care of:

    - removing duplicated timesteps across files,
    - merging/splitting files into daily files,
    - regularizing timesteps for potentially trailing seconds,
    - ensuring L0C files with unique sample intervals.

    Duplicated timesteps are automatically dropped if their variable values coincides,
    otherwise an error is raised.

    This function is intended to be called through the ``disdrodb_run_l0c_station``
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
    remove_l0b: bool, optional
        Whether to remove the processed L0B files. The default value is ``False``.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    """
    # Define product
    product = "L0C"

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
    t_i = time.time()
    msg = f"{product} processing of station {station_name} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------------.
    # Define logs directory
    logs_dir = create_logs_directory(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create product directory
    data_dir = create_product_directory(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        force=force,
    )

    # ------------------------------------------------------------------------.
    # Define metadata filepath
    metadata_filepath = define_metadata_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
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
            f"{product} processing of {data_source} {campaign_name} {station_name} "
            + f"has not been launched because of missing {required_product} data."
        )
        print(msg)
        return

    # -------------------------------------------------------------------------.
    # Retrieve dictionary with the required files for each day.
    dict_days_files = get_files_per_days(filepaths)

    # -----------------------------------------------------------------.
    # Generate L0C files
    # - Loop over the L0 netCDF files and generate L1 files.
    # - If parallel=True, it does that in parallel using dask.delayed
    list_tasks = [
        _generate_l0c(
            day=day,
            filepaths=filepaths,
            data_dir=data_dir,
            logs_dir=logs_dir,
            metadata_filepath=metadata_filepath,
            campaign_name=campaign_name,
            station_name=station_name,
            # Processing options
            force=force,
            verbose=verbose,
            parallel=parallel,
        )
        for day, filepaths in dict_days_files.items()
    ]
    list_logs = dask.compute(*list_tasks) if parallel else list_tasks

    # -----------------------------------------------------------------.
    # Define summary logs
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
    # End processing
    timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
    msg = f"{product} processing of station {station_name} completed in {timedelta_str}"
    log_info(logger=logger, msg=msg, verbose=verbose)

    # -----------------------------------------------------------------.
    # Option to remove L0B
    if remove_l0b:
        remove_product(
            data_archive_dir=data_archive_dir,
            product="L0B",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            logger=logger,
            verbose=verbose,
        )
