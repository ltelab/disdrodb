#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:30:51 2022

@author: ghiggi
"""
import os
import time
import dask
import shutil
import click
import logging
import functools
import datetime
import numpy as np

# Directory
from disdrodb.L0.io import get_raw_file_list, get_l0a_file_list
from disdrodb.L0.io import create_directory_structure_l0a, create_directory_structure

# Metadata
from disdrodb.L0.metadata import read_metadata

# Standards
from disdrodb.L0.check_standards import check_sensor_name

# L0B_processing
from disdrodb.L0.L0B_concat import concatenate_L0B_files
from disdrodb.L0.utils_scripts import _execute_cmd
# Logger
from disdrodb.utils.logger import (
    create_file_logger,
    close_logger,
    define_summary_log,
    log_info,
    # log_warning,
    log_error,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------.
#### Creation of L0A and L0B Single Station File


def _delayed_based_on_kwargs(function):
    """Decorator to make the function delayed if its `parallel` argument is True."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Check if it must be a delayed function
        parallel = kwargs.get("parallel")
        # If parallel is True
        if parallel:
            # Enforce verbose to be False
            kwargs["verbose"] = False
            # Define the delayed task
            result = dask.delayed(function)(*args, **kwargs)
        else:
            # Else run the function
            result = function(*args, **kwargs)
        return result

    return wrapper


@_delayed_based_on_kwargs
def _generate_l0a(
    filepath,
    processed_dir,
    station_id,  # retrievable from filepath
    column_names,
    reader_kwargs,
    df_sanitizer_fun,
    force,
    verbose,
    parallel,
):
    from disdrodb.L0.io import get_L0A_fpath
    from disdrodb.L0.L0A_processing import (
        process_raw_file,
        write_df_to_parquet,
    )

    ##------------------------------------------------------------------------.
    # Create file logger
    filename = os.path.basename(filepath)
    logger = create_file_logger(
        processed_dir=processed_dir,
        product_level="L0A",
        station_id=station_id,
        filename=filename,
        parallel=parallel,
    )
    logger_fpath = logger.handlers[0].baseFilename

    ##------------------------------------------------------------------------.
    # Log start processing
    msg = f"L0A processing of {filename} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    # Retrieve metadata
    attrs = read_metadata(raw_dir=processed_dir, station_id=station_id)

    # Retrieve sensor name
    sensor_name = attrs["sensor_name"]
    check_sensor_name(sensor_name)

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
        )

        ##--------------------------------------------------------------------.
        #### - Write to Parquet
        fpath = get_L0A_fpath(df=df, processed_dir=processed_dir, station_id=station_id)
        write_df_to_parquet(df=df, fpath=fpath, force=force, verbose=verbose)

        ##--------------------------------------------------------------------.
        # Clean environment
        del df

        # Log end processing
        msg = f"L0A processing of {filename} has ended."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # Otherwise log the error
    except Exception as e:
        error_type = str(type(e).__name__)
        msg = f"{error_type}: {e}"
        log_error(logger=logger, msg=msg, verbose=verbose)
        pass

    # Close the file logger
    close_logger(logger)

    # Return the logger file path
    return logger_fpath


@_delayed_based_on_kwargs
def _generate_l0b(
    filepath,
    processed_dir,  # retrievable from filepath
    station_id,  # retrievable from filepath
    force,
    verbose,
    debugging_mode,
    parallel,
):
    from disdrodb.utils.logger import create_file_logger
    from disdrodb.L0.io import get_L0B_fpath, read_L0A_dataframe
    from disdrodb.L0.L0B_processing import (
        create_L0B_from_L0A,
        write_L0B,
    )

    # -----------------------------------------------------------------.
    # Create file logger
    filename = os.path.basename(filepath)
    logger = create_file_logger(
        processed_dir=processed_dir,
        product_level="L0B",
        station_id=station_id,
        filename=filename,
        parallel=parallel,
    )
    logger_fpath = logger.handlers[0].baseFilename

    ##------------------------------------------------------------------------.
    # Log start processing
    msg = f"L0B processing of {filename} has started."
    log_info(logger, msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    # Retrieve metadata
    attrs = read_metadata(raw_dir=processed_dir, station_id=station_id)

    # Retrieve sensor name
    sensor_name = attrs["sensor_name"]
    check_sensor_name(sensor_name)

    ##------------------------------------------------------------------------.
    try:
        # Read L0A Apache Parquet file
        df = read_L0A_dataframe(
            filepath, verbose=verbose, debugging_mode=debugging_mode
        )
        # -----------------------------------------------------------------.
        # Create xarray Dataset
        ds = create_L0B_from_L0A(df=df, attrs=attrs, verbose=verbose)

        # -----------------------------------------------------------------.
        # Write L0B netCDF4 dataset
        fpath = get_L0B_fpath(ds, processed_dir, station_id)
        write_L0B(ds, fpath=fpath, force=force)

        ##--------------------------------------------------------------------.
        # Clean environment
        del ds, df

        # Log end processing
        msg = f"L0B processing of {filename} has ended."
        log_info(logger, msg, verbose=verbose)

    # Otherwise log the error
    except Exception as e:
        error_type = str(type(e).__name__)
        msg = f"{error_type}: {e}"
        log_error(logger, msg, verbose=verbose)
        pass

    # Close the file logger
    close_logger(logger)

    # Return the logger file path
    return logger_fpath


####------------------------------------------------------------------------.
#### Creation of L0A and L0B Single Station Files


def run_l0a(
    raw_dir,
    processed_dir,
    station,
    # L0A reader argument
    files_glob_pattern,
    column_names,
    reader_kwargs,
    df_sanitizer_fun,
    # Processing options
    parallel,
    verbose,
    force,
    debugging_mode,
):
    # ------------------------------------------------------------------------.
    # Start L0A processing
    if verbose:
        t_i = time.time()
        msg = f" - L0A processing of station {station} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------------.
    # Create directory structure 
    create_directory_structure_l0a(raw_dir=raw_dir, 
                                   processed_dir=processed_dir,
                                   station=station, 
                                   force=force)

    # -------------------------------------------------------------------------.
    # List files to process
    filepaths = get_raw_file_list(
        raw_dir=raw_dir,
        station_id=station,
        # L0A reader argument
        glob_patterns=files_glob_pattern,
        # Processing options
        verbose=verbose,
        debugging_mode=debugging_mode,
    )

    # -----------------------------------------------------------------.
    # Generate L0A files
    # - Loop over the files and save the L0A Apache Parquet files.
    # - If parallel=True, it does that in parallel using dask.delayed
    list_tasks = []
    for filepath in filepaths:
        list_tasks.append(
            _generate_l0a(
                filepath=filepath,
                processed_dir=processed_dir,
                station_id=station,  # TODO: remove in future
                # L0A reader argument
                column_names=column_names,
                reader_kwargs=reader_kwargs,
                df_sanitizer_fun=df_sanitizer_fun,
                # Processing options
                force=force,
                verbose=verbose,
                parallel=parallel,
            )
        )
    if parallel:
        list_logs = dask.compute(*list_tasks)
    else:
        list_logs = list_tasks
    # -----------------------------------------------------------------.
    # Define L0A summary logs
    define_summary_log(list_logs)

    # ---------------------------------------------------------------------.
    # End L0A processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
        msg = f" - L0A processing of station {station} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)
    return None


def run_l0b(
    processed_dir,
    station,
    # Processing options
    parallel,
    force,
    verbose,
    debugging_mode,
):
    # -----------------------------------------------------------------.
    # Start L0B processing
    if verbose:
        t_i = time.time()
        msg = f" - L0B processing of station_id {station} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -------------------------------------------------------------------------.
    # Create directory structure 
    create_directory_structure(processed_dir=processed_dir,
                               product_level="L0B",
                               station=station, 
                               force=force)
    
    ##----------------------------------------------------------------.
    # Get L0A files for the station
    filepaths = get_l0a_file_list(
        processed_dir=processed_dir,
        station_id=station,
        debugging_mode=debugging_mode,
    )

    # -----------------------------------------------------------------.
    # Generate L0B files
    # - Loop over the L0A files and save the L0B netCDF files.
    # - If parallel=True, it does that in parallel using dask.delayed
    list_tasks = []
    for filepath in filepaths:
        list_tasks.append(
            _generate_l0b(
                filepath=filepath,
                processed_dir=processed_dir,  # can be derived by filepath
                station_id=station,  # can be derived by filepath
                force=force,
                verbose=verbose,
                debugging_mode=debugging_mode,
                parallel=parallel,
            )
        )
    if parallel:
        list_logs = dask.compute(*list_tasks)
    else:
        list_logs = list_tasks

    # -----------------------------------------------------------------.
    # Define L0B summary logs
    define_summary_log(list_logs)

    # -----------------------------------------------------------------.
    # End L0B processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
        msg = (
            f" - L0B processing of station_id {station} completed in {timedelta_str}"
        )
        log_info(logger=logger, msg=msg, verbose=verbose)
    return None


def run_disdrodb_l0a_station(
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
    """Run the L0B processing of a station calling run_disdrodb_l0a_station in the terminal."""
    # TODO: note that this does not customize the number of dask workers !
   
    # Define command 
    cmd = " ".join(
        [
            "run_disdrodb_l0a_station",
            # Station arguments
            disdrodb_dir,
            data_source,
            campaign_name,
            station,
            # Processing options
            "--force",
            str(force),
            "--verbose",
            str(verbose),
            "--debugging_mode",
            str(debugging_mode),
            "--parallel",
            str(parallel),
        ]
    )
    # Execute command 
    _execute_cmd(cmd)
    return None


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
    """Run the L0B processing of a station calling run_disdrodb_l0b_station in the terminal."""
    # Define command 
    cmd = " ".join(
        [
            "run_disdrodb_l0b_station",
            # Station arguments
            disdrodb_dir,
            data_source,
            campaign_name,
            station,
            # Processing options
            "--force",
            str(force),
            "--verbose",
            str(verbose),
            "--debugging_mode",
            str(debugging_mode),
            "--parallel",
            str(parallel),
        ]
    )
    # Execute command 
    _execute_cmd(cmd)
    return None

####--------------------------------------------------------------------------.
#### Run L0 station processing (L0A + L0B)


def run_disdrodb_l0_station(
    disdrodb_dir,
    data_source,
    campaign_name,
    station,
    # L0A settings
    l0a_processing: bool = True,
    # L0B settings
    l0b_processing: bool = True,
    keep_l0a: bool = False,
    single_netcdf: bool = True,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):
    # ---------------------------------------------------------------------.
    msg = f" - L0 processing of station {station} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------.
    # L0A processing
    if l0a_processing:
        run_disdrodb_l0a_station(
            # Station arguments
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station=station,
            # Processing options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
    # ------------------------------------------------------------------.
    # L0B processing
    if l0b_processing:
        run_disdrodb_l0b_station(
            # Station arguments
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station=station,
            # Processing options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )

    # ------------------------------------------------------------------------.
    # Remove L0A station directory if keep_l0a = False
    if not keep_l0a:
        campaign_dir = _get_disdrodb_directory(
            disdrodb_dir=disdrodb_dir,
            product_level="L0A",
            data_source=data_source,
            campaign_name=campaign_name,
        )
        station_product_dir = os.path.join(campaign_dir, "L0A", station)
        shutil.rmtree(station_product_dir)
    
    # ------------------------------------------------------------------------.
    # If single_netcdf=True, concat the netCDF in a single file
    if single_netcdf:
        concatenate_L0B_station(
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station=station,
            remove=False,   # TODO make as argument  # keep_l0b_files, concatenate_l0b
            verbose=verbose
        )
    return None 
    
    # -------------------------------------------------------------------------.
    # End of L0 processing for all stations
    timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
    msg = (
        f" - L0 processing of stations {station} completed in {timedelta_str}"
    )
    log_info(logger, msg, verbose)
    return None 
 

 


####---------------------------------------------------------------------------.
#### Wrappers to run archive L0 processing
# TODO: create script calling run_disdrodb_l0, run_disdrodb_l0a, run_disdrodb_l0b

def run_disdrodb_l0(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station=None,
    # L0A settings
    l0a_processing: bool = True,
    # L0B settings
    l0b_processing: bool = True,
    keep_l0a: bool = False,
    single_netcdf: bool = True,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):
    from disdrodb.api.io import available_stations

    if l0a_processing:
        product_level = "RAW"
    elif l0b_processing:
        product_level = "L0A"
    else:
        # TODO: potentially can be used to just run single_netcdf
        raise ValueError("At least l0a_processing or l0b_processing must be True.")

    # Get list of available stations
    list_info = available_stations(
        disdrodb_dir,
        product_level=product_level,
        data_sources=data_sources,
        campaign_names=campaign_names,
    )

    # If no stations available, raise an error
    if len(list_info) == 0:
        raise ValueError("No stations are available !")

    # Filter by provided stations
    if station is not None:
        list_info = [info for info in list_info if info[2] in station]
        # If nothing left, raise an error
        if len(list_info) == 0:
            raise ValueError(
                "No stations to concatenate given the provided `station` argument!"
            )

    # Print message
    n_stations = len(list_info)
    print(f"L0 processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station in list_info:
        print(
            f"L0 processing of {data_source} {campaign_name} {station} station started."
        )
        # Run processing
        run_disdrodb_l0_station(
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station=station,
            # L0A settings
            l0a_processing=l0a_processing,
            # L0B settings
            l0b_processing=l0b_processing,
            keep_l0a=keep_l0a,
            single_netcdf=single_netcdf,
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(
            f"L0 processing of {data_source} {campaign_name} {station} station ended."
        )


def run_disdrodb_l0a(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station=None,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):

    run_disdrodb_l0(
        disdrodb_dir=disdrodb_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station=station,
        # L0A settings
        l0a_processing=True,
        # L0B settings
        l0b_processing=False,
        keep_l0a=True,
        single_netcdf=False,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )


def run_disdrodb_l0b(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station=None,
    # L0B settings
    keep_l0a: bool = True,
    single_netcdf: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):

    run_disdrodb_l0(
        disdrodb_dir=disdrodb_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station=station,
        # L0A settings
        l0a_processing=False,
        # L0B settings
        l0b_processing=True,
        keep_l0a=keep_l0a,
        single_netcdf=single_netcdf,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )
    

####--------------------------------------------------------------------------.

#### TO DEPRECATE


def get_station_list_to_process(raw_dir, station_ids):
    """
    Retrieve the subset of stations to process.

    Parameters
    ----------
    raw_dir : str
        The directory path where all the raw content of a specific campaign is stored.
        The path must have the following structure:
            <...>/DISDRODB/Raw/<data_source>/<campaign_name>'.
        Inside the raw_dir directory, it is required to adopt the following structure:
        - /data/<station_id>/<raw_files>
        - /metadata/<station_id>.yaml
        Important points:
        - For each <station_id> there must be a corresponding YAML file in the metadata subfolder.
        - The <campaign_name> must semantically match between:
           - the raw_dir and processed_dir directory paths;
           - with the key 'campaign_name' within the metadata YAML files.
        - The campaign_name are expected to be UPPER CASE.
    station_ids : str or list
        If None (default), it process all the stations inside the raw_dir
        If a single or a list of station_id, it process only those stations.

    Returns
    -------
    list
        List of stations to process.

    """
    # Check station_ids type
    if not isinstance(station_ids, (str, list, type(None))):
        raise TypeError("`station_ids` must be None, str or list of strings.")
    # Select available stations in raw_dir
    data_dir = os.path.join(raw_dir, "data")
    list_stations_id = sorted(os.listdir(data_dir))
    # Check if there are stations
    if len(list_stations_id) == 0:
        raise ValueError(f"No station directories inside {data_dir}")
    # -----------------------------------------.
    # Case 1: station_ids=None
    if station_ids is None:
        return list_stations_id

    # -----------------------------------------.
    # Case 2: list of selected stations provided
    if isinstance(station_ids, str):
        station_ids = [station_ids]
    # - Check stations_id type
    is_not_string = [not isinstance(station_id, str) for station_id in station_ids]
    if np.any(is_not_string):
        raise ValueError("`station_ids` must be specified as (list of) strings.")
    # - Check if specified stations are availables
    idx_not_valid = np.isin(station_ids, list_stations_id, invert=True)
    if np.any(idx_not_valid):
        unvalid_station_ids = np.array(station_ids)[idx_not_valid].tolist()
        raise ValueError(
            f"Valid station_ids are {list_stations_id}. {unvalid_station_ids} are not."
        )
    # - Return station_ids subset
    return station_ids


####--------------------------------------------------------------------------.
#### CLIck


def click_l0_station_arguments(function: object):
    """Click command line arguments for L0 processing of a station.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.argument("station", metavar="<station>")(function)
    function = click.argument("campaign_name", metavar="<campaign_name>")(function)
    function = click.argument("data_source", metavar="<data_source>")(function)
    function = click.argument("disdrodb_dir", metavar="<disdrodb_dir>")(function)
    return function 

 
def click_l0_processing_options(function: object):
    """Click command line default parameters for L0 processing options.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "-p",
        "--parallel",
        type=bool,
        show_default=True,
        default=False,
        help="Process files in parallel",
    )(function)
    function = click.option(
        "-d",
        "--debugging_mode",
        type=bool,
        show_default=True,
        default=False,
        help="Switch to debugging mode",
    )(function)
    function = click.option(
        "-v", "--verbose", type=bool, show_default=True, default=False, help="Verbose"
    )(function)
    function = click.option(
        "-f",
        "--force",
        type=bool,
        show_default=True,
        default=False,
        help="Force overwriting",
    )(function)
    return function


def click_l0_archive_options(function: object):
    """Click command line arguments for L0 processing archiving of a station.
    
    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "-s",
        "--single_netcdf",
        type=bool,
        show_default=True,
        default=True,
        help="Produce single L0B netCDF",
    )(function)
    function = click.option(
        "-k",
        "--keep_l0a",
        type=bool,
        show_default=True,
        default=True,
        help="Whether to keep the L0A Parquet file",
    )(function)
    function = click.option(
        "-l0b",
        "--l0b_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Perform L0B processing",
    )(function)
    function = click.option(
        "-l0a",
        "--l0a_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Perform L0A processing",
    )(function)
    return function
  