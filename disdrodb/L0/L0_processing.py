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
from urllib import response
import click
import logging

# Directory
from disdrodb.L0.io import (
    check_directories,
    get_campaign_name,
    create_directory_structure,
    get_L0A_dir,
    get_L0A_fpath,
    get_L0B_fpath,
    read_L0A_dataframe,
)

# Metadata
from disdrodb.L0.metadata import read_metadata

# Standards
from disdrodb.L0.check_standards import check_sensor_name, check_L0A_standards

# L0A_processing
from disdrodb.L0.L0A_processing import (
    get_file_list,
    read_L0A_raw_file_list,
    write_df_to_parquet,
)

# L0B_processing
from disdrodb.L0.L0B_processing import (
    create_L0B_from_L0A,
    write_L0B,
    create_summary_statistics,
)

# Logger
from disdrodb.utils.logger import create_l0_logger, close_logger
from disdrodb.utils.logger import log_info, log_warning

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------.
# Consistency choice
# TODO:
# - Add verbose and logs to disdrodb.io function !!!

# -----------------------------------------------------------------------------.
def click_L0_readers_options(function: object):
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


# -----------------------------------------------------------------------------.
def run_L0(
    # Arguments custom to each reader
    column_names,
    reader_kwargs,
    files_glob_pattern,
    df_sanitizer_fun,
    raw_dir,
    processed_dir,
    # Arguments designing the processing type
    l0a_processing=True,
    l0b_processing=True,
    keep_l0a=True,
    force=False,
    verbose=False,
    debugging_mode=False,
    lazy=True,
    single_netcdf=True,
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
        Dictionary of arguments to be passed to `pd.read_csv` or `dd.read_csv` to
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

    t_i_script = time.time()
    # -------------------------------------------------------------------------.
    # Initial directory checks
    raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

    # Retrieve campaign name
    campaign_name = get_campaign_name(raw_dir)

    # -------------------------------------------------------------------------.
    # Define logging settings
    logger = create_l0_logger(processed_dir, campaign_name)

    # -------------------------------------------------------------------------.
    # Create directory structure
    create_directory_structure(raw_dir, processed_dir)

    # -------------------------------------------------------------------------.
    #### Loop over station_id directory and process the files
    list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

    # station_id = list_stations_id[0]
    for station_id in list_stations_id:
        # ---------------------------------------------------------------------.
        logger.info(f" - Processing of station_id {station_id} has started")
        # ---------------------------------------------------------------------.
        # Retrieve metadata
        attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)
        # Retrieve sensor name
        sensor_name = attrs["sensor_name"]
        check_sensor_name(sensor_name)

        # ---------------------------------------------------------------------.
        ########################
        #### L0A processing ####
        ########################
        if l0a_processing:
            # Start L0 processing
            t_i_station = time.time()
            msg = " - L0A processing of station_id {} has started.".format(station_id)
            if verbose:
                print(msg)
            logger.info(msg)

            # -----------------------------------------------------------------.
            #### - List files to process
            glob_pattern = os.path.join("data", station_id, files_glob_pattern)
            file_list = get_file_list(
                raw_dir=raw_dir,
                glob_pattern=glob_pattern,
                verbose=verbose,
                debugging_mode=debugging_mode,
            )

            # -----------------------------------------------------------------.
            #### - If single_netcdf = True, ensure loop over all files only once
            if single_netcdf:
                file_list = [file_list]

            # -----------------------------------------------------------------.
            #### - Loop over all files
            # - It loops only once if single_netcdf=True
            for filepath in file_list:
                ##------------------------------------------------------.
                # Define file suffix
                if single_netcdf:
                    file_suffix = ""
                else:
                    # Get file name without file extensions
                    t_i_file = time.time()
                    file_suffix = os.path.basename(filepath).split(".")[0]
                    logger.info(
                        f"L0A processing of raw file {file_suffix} has started."
                    )

                ##------------------------------------------------------.
                #### - Read all raw data files into a dataframe
                df = read_L0A_raw_file_list(
                    file_list=filepath,
                    column_names=column_names,
                    reader_kwargs=reader_kwargs,
                    df_sanitizer_fun=df_sanitizer_fun,
                    lazy=lazy,
                    sensor_name=sensor_name,
                    verbose=verbose,
                )

                ##------------------------------------------------------.
                #### - Write to Parquet
                fpath = get_L0A_fpath(processed_dir, station_id, suffix=file_suffix)
                write_df_to_parquet(df=df, fpath=fpath, force=force, verbose=verbose)

                ##------------------------------------------------------.
                #### - Check L0 file respects the DISDRODB standards
                check_L0A_standards(
                    fpath=fpath, sensor_name=sensor_name, verbose=verbose
                )

                ##------------------------------------------------------.
                # Delete temp variables
                del df

                ##------------------------------------------------------.
                if not single_netcdf:
                    # End L0 processing for a single raw file
                    t_f = time.time() - t_i_file
                    msg = " - L0A processing of {} ended in {:.2f}s".format(
                        file_suffix, t_f
                    )
                    log_info(logger, msg, verbose)

            ##------------------------------------------------------.
            # End L0 processing for the station
            t_f = time.time() - t_i_station
            msg = " - L0A processing of station_id {} ended in {:.2f}s".format(
                station_id, t_f
            )
            log_info(logger, msg, verbose)

        # ------------------------------------------------------------------.
        ########################
        #### L0B processing ####
        ########################
        if l0b_processing:
            # Start L1 processing
            t_i = time.time()
            msg = " - L0B processing of station_id {} has started.".format(station_id)
            if verbose:
                print(msg)
            logger.info(msg)
            ##----------------------------------------------------------------.
            # Get station L0A directory
            L0A_dir_path = get_L0A_dir(processed_dir, station_id)
            file_list = glob.glob(os.path.join(L0A_dir_path, "*.parquet"))
            n_files = len(file_list)
            if n_files == 0:
                msg = f"No L0A Apache Parquet file is available in {L0A_dir_path}. Run L0A processing first."
                logger.error(msg)
                raise ValueError(msg)

            ##----------------------------------------------------------------.
            # Checks for single_netcdf=True
            if single_netcdf:
                # Enclose into a list to loop over only once
                file_list = [file_list]
                if n_files != 1:
                    msg = "If single_netcdf=True, DISDRODB would typically expect only a single L0A Apache Parquet file in {L0A_dir_path}."
                    log_warning(logger, msg, verbose)

            ##----------------------------------------------------------------.
            # Loop over all files
            for filepath in file_list:
                ##------------------------------------------------------.
                # Define file suffix
                if single_netcdf:
                    file_suffix = ""
                else:
                    # Get file name without file extensions
                    t_i_file = time.time()
                    file_suffix = os.path.basename(filepath).split(".")[0]
                    logger.info(
                        f"L0A processing of raw file {file_suffix} has started."
                    )
                ##------------------------------------------------------.
                # Read L0A dataframes
                df = read_L0A_dataframe(
                    filepath, lazy=lazy, verbose=verbose, debugging_mode=debugging_mode
                )

                # -----------------------------------------------------------------.
                #### - Create xarray Dataset
                ds = create_L0B_from_L0A(df=df, attrs=attrs, lazy=lazy, verbose=verbose)

                # -----------------------------------------------------------------.
                #### - Write L0B netCDF4 dataset
                fpath = get_L0B_fpath(ds, processed_dir, station_id)
                write_L0B(ds, fpath=fpath, sensor_name=sensor_name)

                # -----------------------------------------------------------------.
                if not single_netcdf:
                    # End L0B processing for a single L0A file
                    t_f = time.time() - t_i_file
                    msg = " - L0B processing of {} ended in {:.2f}s".format(
                        file_suffix, t_f
                    )
                    log_info(logger, msg, verbose)

            # -----------------------------------------------------------------.
            #### - Compute L0B summary statics (if single_netcdf=True)
            if single_netcdf:
                create_summary_statistics(
                    ds,
                    processed_dir=processed_dir,
                    station_id=station_id,
                    sensor_name=sensor_name,
                )

            # -----------------------------------------------------------------.
            # End L0B processing
            t_f = time.time() - t_i
            msg = " - L0B processing of station_id {} ended in {:.2f}s".format(
                station_id, t_f
            )
            log_info(logger, msg, verbose)

            # -----------------------------------------------------------------.
        # ---------------------------------------------------------------------.
    # Remove L0A directory if keep_L0A = False
    if not keep_l0a:
        shutil.rmtree(os.path.join(processed_dir, "L0A"))

    # -------------------------------------------------------------------------.
    # End of L0B processing for all stations
    t_f = time.time() - t_i_script
    msg = " - L0 processing of stations {} ended in {:.2f} minutes".format(
        list_stations_id, t_f / 60
    )

    # -------------------------------------------------------------------------.
    # Final logs
    logger.info("---")
    msg = "### Script finish ###"
    log_info(logger, msg, verbose)
    close_logger(logger)


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
    Please run get_available_readers() to get the list of all available reader

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
        msg = (
            f"Data source {data_source} has not been found within the available readers"
        )
        logger.exception(msg)
        raise ValueError(msg)

    return correct_data_source


def get_available_readers_by_data_source(data_source: str) -> dict:
    """Return the available readers by data source

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
    Please run get_available_readers() to get the list of all available reader

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
    """Returns the reader function based on input parameters

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


def is_documented_by(original):
    """Wrapper function to apply generic docstring to the decorated function

    Parameters
    ----------
    original : function
        funtion to take the docstring from
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
