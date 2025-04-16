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

from disdrodb.api.checks import check_sensor_name

# Directory
from disdrodb.api.create_directories import (
    create_l0_directory_structure,
    create_logs_directory,
    create_product_directory,
)
from disdrodb.api.info import infer_path_info_tuple
from disdrodb.api.io import find_files, get_required_product, remove_product
from disdrodb.api.path import (
    define_campaign_dir,
    define_l0a_filename,
    define_l0b_filename,
    define_l0c_filename,
    define_metadata_filepath,
)

# get_disdrodb_path,
from disdrodb.configs import get_base_dir
from disdrodb.issue import read_station_issue
from disdrodb.l0.io import (
    get_raw_filepaths,
    read_l0a_dataframe,
)
from disdrodb.l0.l0_reader import get_station_reader_function
from disdrodb.l0.l0a_processing import (
    process_raw_file,
    write_l0a,
)
from disdrodb.l0.l0b_nc_processing import create_l0b_from_raw_nc
from disdrodb.l0.l0b_processing import (
    create_l0b_from_l0a,
    set_l0b_encodings,
    write_l0b,
)
from disdrodb.l0.l0c_processing import (
    create_daily_file,
    get_files_per_days,
    retrieve_possible_measurement_intervals,
)
from disdrodb.metadata import read_station_metadata
from disdrodb.utils.decorator import delayed_if_parallel, single_threaded_if_parallel

# Logger
from disdrodb.utils.logger import (
    close_logger,
    create_logger_file,
    create_product_logs,
    log_error,
    log_info,
)

# log_warning,
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
    # Reader arguments
    column_names,
    reader_kwargs,
    df_sanitizer_fun,
    # Processing info
    sensor_name,
    issue_dict,
    # Processing options
    force,
    verbose,
    parallel,
):
    """Generate L0A file from raw file."""
    # Define product
    product = "L0A"

    ##------------------------------------------------------------------------.
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
    try:
        #### - Read raw file into a dataframe and sanitize to L0A format
        df = process_raw_file(
            filepath=filepath,
            column_names=column_names,
            reader_kwargs=reader_kwargs,
            df_sanitizer_fun=df_sanitizer_fun,
            sensor_name=sensor_name,
            verbose=verbose,
            issue_dict=issue_dict,
        )

        ##--------------------------------------------------------------------.
        #### - Write to Parquet
        filename = define_l0a_filename(df=df, campaign_name=campaign_name, station_name=station_name)
        filepath = os.path.join(data_dir, filename)
        write_l0a(df=df, filepath=filepath, force=force, verbose=verbose)

        ##--------------------------------------------------------------------.
        # Clean environment
        del df

        # Log end processing
        msg = f"{product} processing of {filename} has ended."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # Otherwise log the error
    except Exception as e:
        error_type = str(type(e).__name__)
        msg = f"{error_type}: {e}"
        log_error(logger=logger, msg=msg, verbose=False)

    # Close the file logger
    close_logger(logger)

    # Return the logger file path
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
    log_info(logger, msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    # Retrieve sensor name
    sensor_name = metadata["sensor_name"]
    check_sensor_name(sensor_name)

    ##------------------------------------------------------------------------.
    try:
        # Read L0A Apache Parquet file
        df = read_l0a_dataframe(filepath, verbose=verbose, debugging_mode=debugging_mode)

        # -----------------------------------------------------------------.
        # Create xarray Dataset
        ds = create_l0b_from_l0a(df=df, attrs=metadata, verbose=verbose)

        # -----------------------------------------------------------------.
        # Write L0B netCDF4 dataset
        filename = define_l0b_filename(ds=ds, campaign_name=campaign_name, station_name=station_name)
        filepath = os.path.join(data_dir, filename)
        write_l0b(ds, filepath=filepath, force=force)

        ##--------------------------------------------------------------------.
        # Clean environment
        del ds, df

        # Log end processing
        msg = f"{product} processing of {filename} has ended."
        log_info(logger, msg, verbose=verbose)

    # Otherwise log the error
    except Exception as e:
        error_type = str(type(e).__name__)
        msg = f"{error_type}: {e}"
        log_error(logger, msg, verbose=verbose)

    # Close the file logger
    close_logger(logger)

    # Return the logger file path
    return logger_filepath


def _generate_l0b_from_nc(
    filepath,
    data_dir,
    logs_dir,
    campaign_name,
    station_name,
    # Processing info
    metadata,
    # Reader arguments
    dict_names,
    ds_sanitizer_fun,
    # Processing options
    force,
    verbose,
    parallel,
):
    import xarray as xr  # Load in each process

    # -----------------------------------------------------------------.
    # Define product name
    product = "L0B"

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
    log_info(logger, msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    # Retrieve sensor name
    sensor_name = metadata["sensor_name"]
    check_sensor_name(sensor_name)

    ##------------------------------------------------------------------------.
    try:
        # Open the raw netCDF
        with xr.open_dataset(filepath, decode_timedelta=False, cache=False) as data:
            ds = data.load()

        # Convert to DISDRODB L0 format
        ds = create_l0b_from_raw_nc(
            ds=ds,
            dict_names=dict_names,
            ds_sanitizer_fun=ds_sanitizer_fun,
            sensor_name=sensor_name,
            verbose=verbose,
            attrs=metadata,
        )
        # -----------------------------------------------------------------.
        # Write L0B netCDF4 dataset
        filename = define_l0b_filename(ds=ds, campaign_name=campaign_name, station_name=station_name)
        filepath = os.path.join(data_dir, filename)
        write_l0b(ds, filepath=filepath, force=force)

        ##--------------------------------------------------------------------.
        # Clean environment
        del ds

        # Log end processing
        msg = f"L0B processing of {filename} has ended."
        log_info(logger, msg, verbose=verbose)

    # Otherwise log the error
    except Exception as e:
        error_type = str(type(e).__name__)
        msg = f"{error_type}: {e}"
        log_error(logger, msg, verbose=verbose)

    # Close the file logger
    close_logger(logger)

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
    # -----------------------------------------------------------------.
    # Define product name
    product = "L0C"

    # -----------------------------------------------------------------.
    # Create file logger
    logger, logger_filepath = create_logger_file(
        logs_dir=logs_dir,
        filename=day,
        parallel=parallel,
    )

    ##------------------------------------------------------------------------.
    # Log start processing
    msg = f"{product} processing for {day} has started."
    log_info(logger, msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    ### Core computation
    try:
        # Retrieve measurement_intervals
        # - TODO: in future available from dataset
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

                # Define filepath
                filename = define_l0c_filename(ds, campaign_name=campaign_name, station_name=station_name)
                filepath = os.path.join(data_dir, filename)

                # Write to disk
                write_product(ds, product=product, filepath=filepath, force=force)

        # Clean environment
        del ds

        # Log end processing
        msg = f"{product} processing for {day} has ended."
        log_info(logger, msg, verbose=verbose)

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


####------------------------------------------------------------------------.
#### Creation of L0A and L0B Single Station Files


def run_l0a(
    raw_dir,
    processed_dir,
    station_name,
    # L0A reader argument
    glob_patterns,
    column_names,
    reader_kwargs,
    df_sanitizer_fun,
    # Processing options
    parallel,
    verbose,
    force,
    debugging_mode,
):
    """Run the L0A processing for a specific DISDRODB station.

    This function is called in each reader to convert raw text files into DISDRODB L0A products.

    Parameters
    ----------
    raw_dir : str
        The directory path where all the raw content of a specific campaign is stored.
        The path must have the following structure: ``<...>/DISDRODB/Raw/<DATA_SOURCE>/<CAMPAIGN_NAME>``.
        Inside the ``raw_dir`` directory, it is required to adopt the following structure::

            - ``/data/<station_name>/<raw_files>``
            - ``/metadata/<station_name>.yml``

        **Important points:**

        - For each ``<station_name>``, there must be a corresponding YAML file in the metadata subdirectory.
        - The ``campaign_name`` are expected to be UPPER CASE.
        - The ``<CAMPAIGN_NAME>`` must semantically match between:
            - the ``raw_dir`` and ``processed_dir`` directory paths;
            - with the key ``campaign_name`` within the metadata YAML files.

    processed_dir : str
        The desired directory path for the processed DISDRODB L0A and L0B products.
        The path should have the following structure: ``<...>/DISDRODB/Processed/<DATA_SOURCE>/<CAMPAIGN_NAME>``.
        For testing purposes, this function exceptionally accepts also a directory path simply ending
        with ``<CAMPAIGN_NAME>`` (e.g., ``/tmp/<CAMPAIGN_NAME>``).
    station_name : str
        The name of the station.
    glob_patterns : str
        Glob pattern to search for data files in ``<raw_dir>/data/<station_name>``.
    column_names : list
        Column names of the raw text file.
    reader_kwargs : dict
        Arguments for Pandas ``read_csv`` function to open the text file.
    df_sanitizer_fun : callable, optional
        Sanitizer function to format the DataFrame into DISDRODB L0A standard.
        Default is ``None``.
    parallel : bool, optional
        If ``True``, process the files simultaneously in multiple processes.
        The number of simultaneous processes can be customized using the ``dask.distributed.LocalCluster``.
        If ``False``, process the files sequentially in a single process.
        Default is ``False``.
    verbose : bool, optional
        If ``True``, print detailed processing information to the terminal.
        Default is ``False``.
    force : bool, optional
        If ``True``, overwrite existing data in destination directories.
        If ``False``, raise an error if data already exists in destination directories.
        Default is ``False``.
    debugging_mode : bool, optional
        If ``True``, reduce the amount of data to process.
        Processes only the first 100 rows of 3 raw data files.
        Default is ``False``.

    """
    # Define product name
    product = "L0A"

    # ------------------------------------------------------------------------.
    # Start L0A processing
    if verbose:
        t_i = time.time()
        msg = f"{product} processing of station {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------------.
    # Create directory structure
    data_dir = create_l0_directory_structure(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        product=product,
        station_name=station_name,
        force=force,
    )

    # -------------------------------------------------------------------------.
    # List files to process
    filepaths = get_raw_filepaths(
        raw_dir=raw_dir,
        station_name=station_name,
        # L0A reader argument
        glob_patterns=glob_patterns,
        # Processing options
        verbose=verbose,
        debugging_mode=debugging_mode,
    )

    # -------------------------------------------------------------------------.
    # Retrieve DISDRODB path components
    base_dir, data_source, campaign_name = infer_path_info_tuple(raw_dir)

    # -------------------------------------------------------------------------.
    # Define logs directory
    logs_dir = create_logs_directory(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # -----------------------------------------------------------------.
    # Read issue YAML file
    issue_dict = read_station_issue(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    ##------------------------------------------------------------------------.
    # Read metadata
    metadata = read_station_metadata(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Retrieve sensor name
    sensor_name = metadata["sensor_name"]
    check_sensor_name(sensor_name)

    # -----------------------------------------------------------------.
    # Generate L0A files
    # - Loop over the files and save the L0A Apache Parquet files.
    # - If parallel=True, it does that in parallel using dask.delayed
    list_tasks = [
        _generate_l0a(
            filepath=filepath,
            data_dir=data_dir,
            logs_dir=logs_dir,
            campaign_name=campaign_name,
            station_name=station_name,
            # Reader argument
            column_names=column_names,
            reader_kwargs=reader_kwargs,
            df_sanitizer_fun=df_sanitizer_fun,
            # Processing info
            sensor_name=sensor_name,
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
    # Define L0A summary logs
    create_product_logs(
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        base_dir=base_dir,
        # Logs list
        list_logs=list_logs,
    )

    # ---------------------------------------------------------------------.
    # End L0A processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
        msg = f"L0A processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)


def run_l0b_from_nc(
    raw_dir,
    processed_dir,
    station_name,
    # Reader argument
    glob_patterns,
    dict_names,
    ds_sanitizer_fun,
    # Processing options
    parallel,
    verbose,
    force,
    debugging_mode,
):
    """Run the L0B processing for a specific DISDRODB station with raw netCDFs.

    This function is called in the reader where raw netCDF files must be converted into DISDRODB L0B format.

    Parameters
    ----------
    raw_dir : str
        The directory path where all the raw content of a specific campaign is stored.
        The path must have the following structure: ``<...>/DISDRODB/Raw/<DATA_SOURCE>/<CAMPAIGN_NAME>``.
        Inside the ``raw_dir`` directory, it is required to adopt the following structure::

            - ``/data/<station_name>/<raw_files>``
            - ``/metadata/<station_name>.yml``

        **Important points:**

        - For each ``<station_name>``, there must be a corresponding YAML file in the metadata subdirectory.
        - The ``campaign_name`` are expected to be UPPER CASE.
        - The ``<CAMPAIGN_NAME>`` must semantically match between:
            - the ``raw_dir`` and ``processed_dir`` directory paths;
            - with the key ``campaign_name`` within the metadata YAML files.

    processed_dir : str
        The desired directory path for the processed DISDRODB L0A and L0B products.
        The path should have the following structure: ``<...>/DISDRODB/Processed/<DATA_SOURCE>/<CAMPAIGN_NAME>``.
        For testing purposes, this function exceptionally accepts also a directory path simply ending
        with ``<CAMPAIGN_NAME>`` (e.g., ``/tmp/<CAMPAIGN_NAME>``).

    station_name : str
        The name of the station.

    glob_patterns: str
        Glob pattern to search data files in ``<raw_dir>/data/<station_name>``.
        Example:  ``glob_patterns = "*.nc"``

    dict_names : dict
        Dictionary mapping raw netCDF variables/coordinates/dimension names
        to DISDRODB standards.

     ds_sanitizer_fun : object, optional
        Sanitizer function to format the raw netCDF into DISDRODB L0B standard.

    force : bool, optional
        If ``True``, overwrite existing data in destination directories.
        If ``False``, raise an error if data already exists in destination directories.
        Default is ``False``.

    verbose : bool, optional
        If ``True``, print detailed processing information to the terminal.
        Default is ``True``.

    parallel : bool, optional
        If ``True``, process the files simultaneously in multiple processes.
        The number of simultaneous processes can be customized using the ``dask.distributed.LocalCluster``.
        Ensure that the ``threads_per_worker`` (number of thread per process) is set to 1 to avoid HDF errors.
        Also, ensure to set the ``HDF5_USE_FILE_LOCKING`` environment variable to ``False``.
        If ``False``, process the files sequentially in a single process.
        If ``False``, multi-threading is automatically exploited to speed up I/0 tasks.
        Default is ``False``.

    debugging_mode : bool, optional
        If ``True``, reduce the amount of data to process.
        Only the first 3 raw netCDF files will be processed.
        Default is ``False``.

    """
    # Define product name
    product = "L0B"

    # ------------------------------------------------------------------------.
    # Start L0B NC processing
    if verbose:
        t_i = time.time()
        msg = f"L0B processing of station {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------------.
    # Create directory structure
    data_dir = create_l0_directory_structure(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        product=product,
        station_name=station_name,
        force=force,
    )

    # -------------------------------------------------------------------------.
    # Retrieve DISDRODB path components
    base_dir, data_source, campaign_name = infer_path_info_tuple(processed_dir)

    # Define logs directory
    logs_dir = create_logs_directory(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # -----------------------------------------------------------------.
    # Retrieve metadata
    metadata = read_station_metadata(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # -------------------------------------------------------------------------.
    # List files to process
    filepaths = get_raw_filepaths(
        raw_dir=raw_dir,
        station_name=station_name,
        # Reader argument
        glob_patterns=glob_patterns,
        # Processing options
        verbose=verbose,
        debugging_mode=debugging_mode,
    )

    # -----------------------------------------------------------------.
    # Generate L0B files
    # - Loop over the raw netCDF files and convert it to DISDRODB netCDF format.
    # - If parallel=True, it does that in parallel using dask.bag
    #   Settings npartitions=len(filepaths) enable to wait prior task on a core
    #   finish before starting a new one.
    list_tasks = [
        _generate_l0b_from_nc(
            filepath=filepath,
            data_dir=data_dir,
            logs_dir=logs_dir,
            campaign_name=campaign_name,
            station_name=station_name,
            # Processing info
            metadata=metadata,
            # Reader arguments
            dict_names=dict_names,
            ds_sanitizer_fun=ds_sanitizer_fun,
            # Processing options
            force=force,
            verbose=verbose,
            parallel=parallel,
        )
        for filepath in filepaths
    ]
    list_logs = dask.compute(*list_tasks) if parallel else list_tasks

    # if not parallel:
    #     list_logs = [
    #         _generate_l0b_from_nc(
    #             filepath=filepath,
    #             data_dir=data_dir,
    #             logs_dir=logs_dir,
    #             campaign_name=campaign_name,
    #             station_name=station_name,
    #             # Processing info
    #             metadata=metadata,
    #             # Reader arguments
    #             dict_names=dict_names,
    #             ds_sanitizer_fun=ds_sanitizer_fun,
    #             # Processing options
    #             force=force,
    #             verbose=verbose,
    #             parallel=parallel,
    #         )
    #         for filepath in filepaths
    #     ]
    # else:
    #     bag = db.from_sequence(filepaths, npartitions=len(filepaths))
    #     list_logs = bag.map(
    #         _generate_l0b_from_nc,
    #         data_dir=data_dir,
    #         logs_dir=logs_dir,
    #         campaign_name=campaign_name,
    #         station_name=station_name,
    #         # Processing info
    #         metadata=metadata,
    #         # Reader arguments
    #         dict_names=dict_names,
    #         ds_sanitizer_fun=ds_sanitizer_fun,
    #         # Processing options
    #         force=force,
    #         verbose=verbose,
    #         parallel=parallel,
    #     ).compute()

    # -----------------------------------------------------------------.
    # Define L0B summary logs
    create_product_logs(
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        base_dir=base_dir,
        # Logs list
        list_logs=list_logs,
    )

    # ---------------------------------------------------------------------.
    # End L0B processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
        msg = f"L0B processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)


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
    base_dir: Optional[str] = None,
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
        Only the first 3 raw data files will be processed. By default, ``False``.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    """
    # Define base directory
    base_dir = get_base_dir(base_dir)

    # Retrieve reader
    reader = get_station_reader_function(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Define campaign raw_dir and process_dir
    raw_dir = define_campaign_dir(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
    )
    processed_dir = define_campaign_dir(
        base_dir=base_dir,
        product="L0A",  # also works for raw netCDFs
        data_source=data_source,
        campaign_name=campaign_name,
    )
    # Run L0A processing
    # --> The reader calls the run_l0a or the run_l0b_from_nc if the raw data are
    # text files or netCDF files respectively.
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
    base_dir: Optional[str] = None,
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
        Only the first 100 rows of 3 L0A files will be processed. The default is ``False``.
    remove_l0a: bool, optional
        Whether to remove the processed L0A files. The default is ``False``.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    """
    # Define product name
    product = "L0B"

    # Retrieve DISDRODB base directory
    base_dir = get_base_dir(base_dir)

    # -----------------------------------------------------------------.
    # Retrieve metadata
    metadata = read_station_metadata(
        base_dir=base_dir,
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
    if verbose:
        t_i = time.time()
        msg = f"{product} processing of station_name {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # Define logs directory
    logs_dir = create_logs_directory(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # -------------------------------------------------------------------------.
    # Create product directory
    data_dir = create_product_directory(
        base_dir=base_dir,
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
            base_dir=base_dir,
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
            f"{product} processing of {data_source} {campaign_name} {station_name}"
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
        base_dir=base_dir,
        # Logs list
        list_logs=list_logs,
    )

    # -----------------------------------------------------------------.
    # End L0B processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
        msg = f"{product} processing of station_name {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -----------------------------------------------------------------.
    # Option to remove L0A
    if remove_l0a:
        remove_product(
            base_dir=base_dir,
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
    base_dir: Optional[str] = None,
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
        Only the first 3 files will be processed. By default, ``False``.
    remove_l0b: bool, optional
        Whether to remove the processed L0B files. The default is ``False``.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    """
    # Define product
    product = "L0C"

    # Define base directory
    base_dir = get_base_dir(base_dir)

    # Define logs directory
    logs_dir = create_logs_directory(
        product=product,
        base_dir=base_dir,
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
    # Create product directory
    data_dir = create_product_directory(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        force=force,
    )

    # ------------------------------------------------------------------------.
    # Define metadata filepath
    metadata_filepath = define_metadata_filepath(
        base_dir=base_dir,
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
            base_dir=base_dir,
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
        base_dir=base_dir,
        # Logs list
        list_logs=list_logs,
    )

    # ---------------------------------------------------------------------.
    # End processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
        msg = f"{product} processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -----------------------------------------------------------------.
    # Option to remove L0B
    if remove_l0b:
        remove_product(
            base_dir=base_dir,
            product="L0B",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            logger=logger,
            verbose=verbose,
        )
