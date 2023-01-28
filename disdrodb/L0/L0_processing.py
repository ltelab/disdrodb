#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:30:51 2022

@author: ghiggi
"""
import os
import time
import glob
import shutil
import click
import logging
import functools
import dask

# Directory
from disdrodb.L0.io import (
    check_directories,
    create_directory_structure,
)

# Metadata
from disdrodb.L0.metadata import read_metadata

# Standards
from disdrodb.L0.check_standards import check_sensor_name

from disdrodb.L0.io import get_raw_file_list, get_l0a_file_list

# L0B_processing
from disdrodb.L0.L0B_processing import create_L0B_archive

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
#### L0A and L0B File Creators


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
    from disdrodb.L0.check_standards import check_L0A_standards
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
    log_info(logger, msg, verbose=verbose)

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
        #### - Check L0 file respects the DISDRODB standards
        check_L0A_standards(fpath=fpath, sensor_name=sensor_name, verbose=verbose)

        ##--------------------------------------------------------------------.
        # Clean environment
        del df

        # Log end processing
        msg = f"L0A processing of {filename} has ended."
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
        write_L0B(ds, fpath=fpath)

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
#### L0, L0A and L0B Routines


def run_l0a(
    raw_dir,
    processed_dir,
    station_id,
    files_glob_pattern,
    column_names,
    reader_kwargs,
    df_sanitizer_fun,
    parallel,
    verbose,
    force,
    debugging_mode,
):
    # ---------------------------------------------------------------------.
    # Start L0A processing
    msg = " - L0A processing of station_id {} has started.".format(station_id)
    if verbose:
        print(msg)

    # -----------------------------------------------------------------.
    # List files to process
    filepaths = get_raw_file_list(
        raw_dir=raw_dir,
        station_id=station_id,
        glob_patterns=files_glob_pattern,
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
                station_id=station_id,
                column_names=column_names,
                reader_kwargs=reader_kwargs,
                df_sanitizer_fun=df_sanitizer_fun,
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
    return None


# -----------------------------------------------------------------.


def run_l0b(
    processed_dir,
    station_id,
    parallel,
    force,
    verbose,
    debugging_mode,
):
    # -----------------------------------------------------------------.
    # Start L0B processing
    if verbose:
        t_i = time.time()
        msg = " - L0B processing of station_id {} has started.".format(station_id)
        print(msg)

    ##----------------------------------------------------------------.
    # Get L0A files for the station
    filepaths = get_l0a_file_list(
        processed_dir=processed_dir,
        station_id=station_id,
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
                station_id=station_id,  # can be derived by filepath
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
        t_f = time.time() - t_i
        msg = " - L0B processing of station_id {} ended in {:.2f}s".format(
            station_id, t_f
        )
    return None


# -----------------------------------------------------------------------------.
def run_L0(
    # Arguments custom to each reader
    column_names,
    reader_kwargs,
    files_glob_pattern,
    df_sanitizer_fun,
    # Arguments designing the processing type
    raw_dir,
    processed_dir,
    l0a_processing=True,
    l0b_processing=True,
    keep_l0a=True,
    single_netcdf=False,
    parallel=False,
    force=False,
    verbose=False,
    debugging_mode=False,
    lazy=False,  # TODO: backcomp
):
    """Core function to process raw data files to L0A and L0B format.

    Parameters
    ----------

    column_names : list
        Column names to be assigned to the dataframe created after reading a raw file.
        This argument is allowed to contains column names not agreeing with the
        DISDRODB standard. However, it's important that non-standard columns are
        dropped during the application of the `df_sanitizer_fun` function.
    reader_kwargs : dict
        Dictionary of arguments to be passed to `pd.read_csv` to
        tailor the reading of the raw file.
    files_glob_pattern: str
        It indicates the glob pattern to search for the raw data files within
        the directory path <raw_dir>/data/<station_id>.
    df_sanitizer_fun : function
        It's a function detailing the custom preprocessing of each single raw data files
        in order to meet the DISDRODB standards.
        The function must have the following definition and return a dataframe:
            def df_sanitizer_fun(df, lazy=False)
                # Import dask or pandas depending on lazy argument
                if lazy:
                    import dask.dataframe as dd
                else:
                    import pandas as dd
                # Custom processing
                pass
                # Return the dataframe agreeing with the DISDRODB standards
                return df
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
    processed_dir : str
        The desired directory path for the processed DISDRODB L0A and L0B products.
        The path should have the following structure:
            <...>/DISDRODB/Processed/<data_source>/<campaign_name>'
        For testing purpose, this function exceptionally accept also a directory path simply ending
        with <campaign_name> (i.e. /tmp/<campaign_name>).
    l0a_processing : bool
      Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
      The default is True.
    l0b_processing : bool
      Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
      The default is True.
    keep_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default is False.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        - For L0A processing, it processes just 3 raw data files.
        - For L0B processing, it takes a small subset of the L0A Apache Parquet dataframe.
        The default is False.
    single_netcdf : bool
        Whether to concatenate all raw files into a single L0B netCDF file.
        If single_netcdf=True, all raw files will be saved into a single L0B netCDF file.
        If single_netcdf=False, each raw file will be converted into the corresponding L0B netCDF file.
        The default is True.

    """

    t_i_script = time.time()
    # -------------------------------------------------------------------------.
    # Initial directory checks
    raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

    # -------------------------------------------------------------------------.
    # Create directory structure
    create_directory_structure(raw_dir, processed_dir)

    # -------------------------------------------------------------------------.
    # Loop over station_id directory and process the files
    list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

    # station_id = list_stations_id[0]
    for station_id in list_stations_id:
        # ---------------------------------------------------------------------.
        logger.info(f" - Processing of station_id {station_id} has started.")

        # ------------------------------------------------------------------.
        # L0A processing
        if l0a_processing:
            run_l0a(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                station_id=station_id,
                files_glob_pattern=files_glob_pattern,
                column_names=column_names,
                reader_kwargs=reader_kwargs,
                df_sanitizer_fun=df_sanitizer_fun,
                parallel=parallel,
                force=force,
                verbose=verbose,
                debugging_mode=debugging_mode,
            )

        # ------------------------------------------------------------------.
        # L0B processing
        if l0b_processing:
            run_l0b(
                processed_dir=processed_dir,
                station_id=station_id,
                parallel=parallel,
                force=force,
                verbose=verbose,
                debugging_mode=debugging_mode,
            )

    # ------------------------------------------------------------------------.
    # Remove L0A directory if keep_l0a = False
    if not keep_l0a:
        shutil.rmtree(os.path.join(processed_dir, "L0A"))

    # ------------------------------------------------------------------------.
    # If single_netcdf=True, concat the netCDF in a single file and compute summary statistics
    if single_netcdf:
        create_L0B_archive(
            processed_dir=processed_dir, station_id=station_id, remove=False
        )

    # -------------------------------------------------------------------------.
    # End of L0 processing for all stations
    t_f = time.time() - t_i_script
    msg = " - L0 processing of stations {} ended in {:.2f} minutes.".format(
        list_stations_id, t_f / 60
    )

    # -------------------------------------------------------------------------.
    # Final logs
    logger.info("---")
    msg = "### Script finish ###"
    log_info(logger, msg, verbose)
    close_logger(logger)


####--------------------------------------------------------------------------.


def get_available_readers() -> dict:
    """Returns the readers description included into the current release of DISDRODB.

    Returns
    -------
    dict
        The dictionary has the following schema {"data_source":{"campaign_name":"reader file path"}}
    """
    # current file path
    lo_folder_path = os.path.dirname(__file__)

    # readers folder path
    reader_folder_path = os.path.join(lo_folder_path, "readers")

    # list of readers folder
    list_of_reader_folder = [
        f.path for f in os.scandir(reader_folder_path) if f.is_dir()
    ]

    # create dictionary
    dict_reader = {}
    for path_folder in list_of_reader_folder:
        data_source = os.path.basename(path_folder)
        dict_reader[data_source] = {}
        for path_python_file in [
            f.path
            for f in os.scandir(path_folder)
            if f.is_file() and f.path.endswith(".py")
        ]:
            reader_name = (
                os.path.basename(path_python_file)
                .replace("reader_", "")
                .replace(".py", "")
            )
            dict_reader[data_source][reader_name] = path_python_file

    return dict_reader


def check_data_source(data_source: str) -> str:
    """Check if the provided data source exists within the available readers.

    Please run get_available_readers() to get the list of all available reader.

    Parameters
    ----------
    data_source : str
        Data source name  - Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)

    Returns
    -------
    str
        if data source exists : retrurn the correct data source name
        if data source does not exist : error

    Raises
    ------
    ValueError
        Error if the data source name provided has not been found.
    """

    dict_all_readers = get_available_readers()

    correct_data_source_list = list(
        set(dict_all_readers.keys()).intersection([data_source, data_source.upper()])
    )

    if correct_data_source_list:
        correct_data_source = correct_data_source_list[0]
    else:
        msg = f"Data source {data_source} has not been found within the available readers."
        logger.exception(msg)
        raise ValueError(msg)

    return correct_data_source


def get_available_readers_by_data_source(data_source: str) -> dict:
    """Return the available readers by data source.

    Parameters
    ----------
    data_source : str
        Data source name - Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)

    Returns
    -------
    dict
        Dictionary that conatins the campaigns for the requested data source.

    """

    correct_data_source = check_data_source(data_source)

    if correct_data_source:
        dict_data_source = get_available_readers().get(correct_data_source)

    return dict_data_source


def check_reader_name(data_source: str, reader_name: str) -> str:
    """Check if the provided data source exists and reader names exists within the available readers.

    Please run get_available_readers() to get the list of all available reader.

    Parameters
    ----------
    data_source : str
        Data source name - Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)
    reader_name : str
        Campaign name

    Returns
    -------
    str
        If True : returns the reader name
        If False : Error - return None

    Raises
    ------
    ValueError
        Error if the reader name provided for the campaign has not been found.
    """

    correct_data_source = check_data_source(data_source)

    if correct_data_source:
        dict_reader_names = get_available_readers_by_data_source(correct_data_source)

        correct_reader_name_list = list(
            set(dict_reader_names.keys()).intersection(
                [reader_name, reader_name.upper()]
            )
        )

        if correct_reader_name_list:
            correct_reader_name = correct_reader_name_list[0]
        else:
            msg = (
                f"Reader {reader_name} has not been found within the available readers"
            )
            logger.exception(msg)
            raise ValueError(msg)

    return correct_reader_name


def get_reader(data_source: str, reader_name: str) -> object:
    """Returns the reader function based on input parameters.

    Parameters
    ----------
    data_source : str
        Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)
    reader_name : str
        Campaign name

    Returns
    -------
    object
        The reader() function

    """

    corrcet_data_source = check_data_source(data_source)
    correct_reader_name = check_reader_name(data_source, reader_name)

    if correct_reader_name:
        full_name = (
            f"disdrodb.L0.readers.{corrcet_data_source}.{correct_reader_name}.reader"
        )
        module_name, unit_name = full_name.rsplit(".", 1)
        my_reader = getattr(__import__(module_name, fromlist=[""]), unit_name)

    return my_reader


####--------------------------------------------------------------------------.


def run_reader(
    data_source: str,
    reader_name: str,
    raw_dir: str,
    processed_dir: str,
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    keep_l0a: bool = False,
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    lazy: bool = True,
    single_netcdf: bool = True,
) -> None:
    """Wrapper to run reader functions.

    Parameters
    ----------
    data_source : str
        Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)
    reader_name : str
        Campaign name
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
    processed_dir : str
        The desired directory path for the processed DISDRODB L0A and L0B products.
        The path should have the following structure:
            <...>/DISDRODB/Processed/<data_source>/<campaign_name>'
        For testing purpose, this function exceptionally accept also a directory path simply ending
        with <campaign_name> (i.e. /tmp/<campaign_name>).
    l0a_processing : bool
      Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
      The default is True.
    l0b_processing : bool
      Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
      The default is True.
    keep_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default is False.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        - For L0A processing, it processes just 3 raw data files.
        - For L0B processing, it takes a small subset of the L0A Apache Parquet dataframe.
        The default is False.
    lazy : bool
        Whether to perform processing lazily with dask.
        If lazy=True, it employed dask.array and dask.dataframe.
        If lazy=False, it employed pandas.DataFrame and numpy.array.
        The default is True.
    single_netcdf : bool
        Whether to concatenate all raw files into a single L0B netCDF file.
        If single_netcdf=True, all raw files will be saved into a single L0B netCDF file.
        If single_netcdf=False, each raw file will be converted into the corresponding L0B netCDF file.
        The default is True.
    """

    # Get the corresponding reader function.
    reader = get_reader(data_source, reader_name)

    # Run the reader
    reader(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        keep_l0a=keep_l0a,
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        lazy=lazy,
        single_netcdf=single_netcdf,
    )


####--------------------------------------------------------------------------.


def is_documented_by(original):
    """Wrapper function to apply generic docstring to the decorated function.

    Parameters
    ----------
    original : function
        Function to take the docstring from.
    """

    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def reader_generic_docstring():
    """Script to process raw data to L0A and L0B format.

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
    processed_dir : str
        The desired directory path for the processed DISDRODB L0A and L0B products.
        The path should have the following structure:
            <...>/DISDRODB/Processed/<data_source>/<campaign_name>'
        For testing purpose, this function exceptionally accept also a directory path simply ending
        with <campaign_name> (i.e. /tmp/<campaign_name>).
    l0a_processing : bool
      Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
      The default is True.
    l0b_processing : bool
      Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
      The default is True.
    keep_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default is False.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        - For L0A processing, it processes just 3 raw data files.
        - For L0B processing, it takes a small subset of the L0A Apache Parquet dataframe.
        The default is False.
    lazy : bool
        Whether to perform processing lazily with dask.
        If lazy=True, it employed dask.array and dask.dataframe.
        If lazy=False, it employed pandas.DataFrame and numpy.array.
        The default is True.
    single_netcdf : bool
        Whether to concatenate all raw files into a single L0B netCDF file.
        If single_netcdf=True, all raw files will be saved into a single L0B netCDF file.
        If single_netcdf=False, each raw file will be converted into the corresponding L0B netCDF file.
        The default is True.
    """


def click_l0_readers_options(function: object):
    """Define click command line parameters.

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
        help="Produce single netCDF",
    )(function)
    function = click.option(
        "-l",
        "--lazy",
        type=bool,
        show_default=True,
        default=True,
        help="Use dask if lazy=True",
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
    function = click.argument("processed_dir", metavar="<processed_dir>")(function)
    function = click.argument(
        "raw_dir", type=click.Path(exists=True), metavar="<raw_dir>"
    )(function)

    return function
