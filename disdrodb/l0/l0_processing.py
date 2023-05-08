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
import os
import time
import dask
import shutil
import click
import logging
import functools
import datetime
import dask.bag as db

# Directory
from disdrodb.l0.io import get_raw_file_list, get_l0a_file_list
from disdrodb.l0.io import (
    create_initial_directory_structure,
    create_directory_structure,
)

# Metadata & Issue
from disdrodb.l0.metadata import read_metadata
from disdrodb.l0.issue import read_issue

# Standards
from disdrodb.l0.check_standards import check_sensor_name

# L0B_processing
from disdrodb.utils.scripts import _execute_cmd

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
    station_name,  # retrievable from filepath
    column_names,
    reader_kwargs,
    df_sanitizer_fun,
    force,
    verbose,
    parallel,
    issue_dict={},
):
    """Generate L0A file from raw file."""

    from disdrodb.l0.io import get_L0A_fpath
    from disdrodb.l0.l0a_processing import (
        process_raw_file,
        write_l0a,
    )

    ##------------------------------------------------------------------------.
    # Create file logger
    filename = os.path.basename(filepath)
    logger = create_file_logger(
        processed_dir=processed_dir,
        product_level="L0A",
        station_name=station_name,
        filename=filename,
        parallel=parallel,
    )

    if not os.environ.get("PYTEST_CURRENT_TEST"):
        logger_fpath = logger.handlers[0].baseFilename
    else:
        logger_fpath = None

    ##------------------------------------------------------------------------.
    # Log start processing
    msg = f"L0A processing of {filename} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    # Retrieve metadata
    attrs = read_metadata(campaign_dir=processed_dir, station_name=station_name)

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
            issue_dict=issue_dict,
        )

        ##--------------------------------------------------------------------.
        #### - Write to Parquet
        fpath = get_L0A_fpath(df=df, processed_dir=processed_dir, station_name=station_name)
        write_l0a(df=df, fpath=fpath, force=force, verbose=verbose)

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
        log_error(logger=logger, msg=msg, verbose=False)
        pass

    # Close the file logger
    close_logger(logger)

    # Return the logger file path
    return logger_fpath


def _generate_l0b(
    filepath,
    processed_dir,  # retrievable from filepath
    station_name,  # retrievable from filepath
    force,
    verbose,
    debugging_mode,
    parallel,
):
    from disdrodb.utils.logger import create_file_logger
    from disdrodb.l0.io import get_L0B_fpath, read_L0A_dataframe
    from disdrodb.l0.l0b_processing import (
        create_l0b_from_l0a,
        write_l0b,
    )

    # -----------------------------------------------------------------.
    # Create file logger
    filename = os.path.basename(filepath)
    logger = create_file_logger(
        processed_dir=processed_dir,
        product_level="L0B",
        station_name=station_name,
        filename=filename,
        parallel=parallel,
    )
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        logger_fpath = logger.handlers[0].baseFilename
    else:
        logger_fpath = None

    ##------------------------------------------------------------------------.
    # Log start processing
    msg = f"L0B processing of {filename} has started."
    log_info(logger, msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    # Retrieve metadata
    attrs = read_metadata(campaign_dir=processed_dir, station_name=station_name)

    # Retrieve sensor name
    sensor_name = attrs["sensor_name"]
    check_sensor_name(sensor_name)

    ##------------------------------------------------------------------------.
    try:
        # Read L0A Apache Parquet file
        df = read_L0A_dataframe(filepath, verbose=verbose, debugging_mode=debugging_mode)
        # -----------------------------------------------------------------.
        # Create xarray Dataset
        ds = create_l0b_from_l0a(df=df, attrs=attrs, verbose=verbose)

        # -----------------------------------------------------------------.
        # Write L0B netCDF4 dataset
        fpath = get_L0B_fpath(ds, processed_dir, station_name)
        write_l0b(ds, fpath=fpath, force=force)

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


def _generate_l0b_from_nc(
    filepath,
    processed_dir,
    station_name,  # retrievable from filepath
    dict_names,
    ds_sanitizer_fun,
    force,
    verbose,
    parallel,
):
    from disdrodb.utils.logger import create_file_logger
    from disdrodb.l0.io import get_L0B_fpath
    from disdrodb.l0.l0b_processing import process_raw_nc, write_l0b

    # -----------------------------------------------------------------.
    # Create file logger
    filename = os.path.basename(filepath)
    logger = create_file_logger(
        processed_dir=processed_dir,
        product_level="L0B",
        station_name=station_name,
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
    attrs = read_metadata(campaign_dir=processed_dir, station_name=station_name)

    # Retrieve sensor name
    sensor_name = attrs["sensor_name"]
    check_sensor_name(sensor_name)

    ##------------------------------------------------------------------------.
    try:
        # Read the raw netCDF and convert to DISDRODB format
        ds = process_raw_nc(
            filepath=filepath,
            dict_names=dict_names,
            ds_sanitizer_fun=ds_sanitizer_fun,
            sensor_name=sensor_name,
            verbose=verbose,
            attrs=attrs,
        )
        # -----------------------------------------------------------------.
        # Write L0B netCDF4 dataset
        fpath = get_L0B_fpath(ds, processed_dir, station_name)
        write_l0b(ds, fpath=fpath, force=force)

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

    Parameters
    ----------
    raw_dir : str
        The directory path where all the raw content of a specific campaign is stored.
        The path must have the following structure:
            <...>/DISDRODB/Raw/<data_source>/<campaign_name>'.
        Inside the raw_dir directory, it is required to adopt the following structure:
        - /data/<station_name>/<raw_files>
        - /metadata/<station_name>.yaml
        Important points:
        - For each <station_name> there must be a corresponding YAML file in the metadata subfolder.
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
    station_name : str
        Station name
    glob_patterns: str
        Glob pattern to search data files in <raw_dir>/data/<station_name>
    column_names : list
        Columns names of the raw text file.
    reader_kwargs : dict
         Pandas `read_csv` arguments to open the text file.
    df_sanitizer_fun : object, optional
        Sanitizer function to format the datafame into DISDRODB L0A standard.
    parallel : bool
        If True, the files are processed simultanously in multiple processes.
        The number of simultaneous processes can be customized using the dask.distributed LocalCluster.
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        It processes just the first 100 rows of 3 raw data files.
        The default is False.
    """

    # ------------------------------------------------------------------------.
    # Start L0A processing
    if verbose:
        t_i = time.time()
        msg = f"L0A processing of station {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------------.
    # Create directory structure
    create_initial_directory_structure(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        product_level="L0A",
        station_name=station_name,
        force=force,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------.
    # List files to process
    filepaths = get_raw_file_list(
        raw_dir=raw_dir,
        station_name=station_name,
        # L0A reader argument
        glob_patterns=glob_patterns,
        # Processing options
        verbose=verbose,
        debugging_mode=debugging_mode,
    )

    # -----------------------------------------------------------------.
    # Read issue YAML file
    issue_dict = read_issue(raw_dir=raw_dir, station_name=station_name)

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
                station_name=station_name,
                # L0A reader argument
                column_names=column_names,
                reader_kwargs=reader_kwargs,
                df_sanitizer_fun=df_sanitizer_fun,
                issue_dict=issue_dict,
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
        msg = f"L0A processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)
    return None


def run_l0b(
    processed_dir,
    station_name,
    # Processing options
    parallel,
    force,
    verbose,
    debugging_mode,
):
    """Run the L0B processing for a specific DISDRODB station.

    Parameters
    ----------
    raw_dir : str
        The directory path where all the raw content of a specific campaign is stored.
        The path must have the following structure:
            <...>/DISDRODB/Raw/<data_source>/<campaign_name>'.
        Inside the raw_dir directory, it is required to adopt the following structure:
        - /data/<station_name>/<raw_files>
        - /metadata/<station_name>.yaml
        Important points:
        - For each <station_name> there must be a corresponding YAML file in the metadata subfolder.
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
        The number of simultaneous processes can be customized using the dask.distributed LocalCluster.
        Ensure that the threads_per_worker (number of thread per process) is set to 1 to avoid HDF errors.
        Also ensure to set the HDF5_USE_FILE_LOCKING environment variable to False.
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        It processes just 3 raw data files.
        The default is False.
    """
    # -----------------------------------------------------------------.
    # Retrieve metadata
    attrs = read_metadata(campaign_dir=processed_dir, station_name=station_name)

    # Skip run_l0b processing if the raw data are netCDFs
    if attrs["raw_data_format"] == "netcdf":
        return None

    # -----------------------------------------------------------------.
    # Start L0B processing
    if verbose:
        t_i = time.time()
        msg = f"L0B processing of station_name {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -------------------------------------------------------------------------.
    # Create directory structure
    create_directory_structure(
        processed_dir=processed_dir,
        product_level="L0B",
        station_name=station_name,
        force=force,
        verbose=verbose,
    )

    ##----------------------------------------------------------------.
    # Get L0A files for the station
    filepaths = get_l0a_file_list(
        processed_dir=processed_dir,
        station_name=station_name,
        debugging_mode=debugging_mode,
    )

    # -----------------------------------------------------------------.
    # Generate L0B files
    # Loop over the L0A files and save the L0B netCDF files.
    # - If parallel=True, it does that in parallel using dask.bag
    #   Settings npartitions=len(filepaths) enable to wait prior task on a core
    #   finish before starting a new one.
    if not parallel:
        list_logs = []
        for filepath in filepaths:
            list_logs.append(
                _generate_l0b(
                    filepath=filepath,
                    processed_dir=processed_dir,
                    station_name=station_name,
                    force=force,
                    verbose=verbose,
                    debugging_mode=debugging_mode,
                    parallel=parallel,
                )
            )
    else:
        bag = db.from_sequence(filepaths, npartitions=len(filepaths))
        list_logs = bag.map(
            _generate_l0b,
            processed_dir=processed_dir,
            station_name=station_name,
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        ).compute()

    # -----------------------------------------------------------------.
    # Define L0B summary logs
    define_summary_log(list_logs)

    # -----------------------------------------------------------------.
    # End L0B processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
        msg = f"L0B processing of station_name {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)
    return None


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

    Parameters
    ----------
    raw_dir : str
        The directory path where all the raw content of a specific campaign is stored.
        The path must have the following structure:
            <...>/DISDRODB/Raw/<data_source>/<campaign_name>'.
        Inside the raw_dir directory, it is required to adopt the following structure:
        - /data/<station_name>/<raw_files>
        - /metadata/<station_name>.yaml
        Important points:
        - For each <station_name> there must be a corresponding YAML file in the metadata subfolder.
        - The <campaign_name> must semantically match between:
           - the raw_dir and processed_dir directory paths;
           - with the key 'campaign_name' within the metadata YAML files.
        - The campaign_name are expected to be UPPER CASE.
    processed_dir : str
        The desired directory path for the processed DISDRODB L0B products.
        The path should have the following structure:
            <...>/DISDRODB/Processed/<data_source>/<campaign_name>'
        For testing purpose, this function exceptionally accept also a directory path simply ending
        with <campaign_name> (i.e. /tmp/<campaign_name>).
    station_name : str
        Station name
    glob_patterns: str
        Glob pattern to search data files in <raw_dir>/data/<station_name>.
        Example: glob_patterns = "*.nc"
    dict_names : dict
        Dictionary mapping raw netCDF variables/coordinates/dimension names
        to DISDRODB standards.
    ds_sanitizer_fun : object, optional
        Sanitizer function to format the raw netCDF into DISDRODB L0B standard.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    parallel : bool
        If True, the files are processed simultanously in multiple processes.
        The number of simultaneous processes can be customized using the dask.distributed LocalCluster.
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        It processes just the first 3 raw netCDF files.
        The default is False.
    """

    # ------------------------------------------------------------------------.
    # Start L0A processing
    if verbose:
        t_i = time.time()
        msg = f"L0B processing of station {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------------.
    # Create directory structure
    create_initial_directory_structure(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        product_level="L0B",
        station_name=station_name,
        force=force,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------.
    # List files to process
    filepaths = get_raw_file_list(
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
    if not parallel:
        list_logs = []
        for filepath in filepaths:
            list_logs.append(
                _generate_l0b_from_nc(
                    filepath=filepath,
                    processed_dir=processed_dir,
                    station_name=station_name,
                    # Reader arguments
                    dict_names=dict_names,
                    ds_sanitizer_fun=ds_sanitizer_fun,
                    # Processing options
                    force=force,
                    verbose=verbose,
                    parallel=parallel,
                )
            )
    else:
        bag = db.from_sequence(filepaths, npartitions=len(filepaths))
        list_logs = bag.map(
            _generate_l0b_from_nc,
            processed_dir=processed_dir,
            station_name=station_name,
            # Reader arguments
            dict_names=dict_names,
            ds_sanitizer_fun=ds_sanitizer_fun,
            # Processing options
            force=force,
            verbose=verbose,
            parallel=parallel,
        ).compute()

    # -----------------------------------------------------------------.
    # Define L0B summary logs
    define_summary_log(list_logs)

    # ---------------------------------------------------------------------.
    # End L0B processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
        msg = f"L0B processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)
    return None


####--------------------------------------------------------------------------.
#### Wrapper to call from terminal


def run_disdrodb_l0a_station(
    # Station arguments
    disdrodb_dir,
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):
    """Run the L0A processing of a station calling run_disdrodb_l0a_station in the terminal."""
    # Define command
    cmd = " ".join(
        [
            "run_disdrodb_l0a_station",
            # Station arguments
            disdrodb_dir,
            data_source,
            campaign_name,
            station_name,
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
    station_name,
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
            station_name,
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
    station_name,
    # L0 archive options
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    l0b_concat: bool = True,
    remove_l0a: bool = False,
    remove_l0b: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
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
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files for each station.
        For L0B, it processes just the first 100 rows of 3 L0A files for each station.
        The default is False.
    """
    from disdrodb.l0.l0b_concat import run_disdrodb_l0b_concat_station
    from disdrodb.api.io import _get_disdrodb_directory

    # ---------------------------------------------------------------------.
    t_i = time.time()
    msg = f"L0 processing of station {station_name} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------.
    # L0A processing
    if l0a_processing:
        run_disdrodb_l0a_station(
            # Station arguments
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
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
            station_name=station_name,
            # Processing options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )

    # ------------------------------------------------------------------------.
    # Remove L0A station directory if remove_l0a = True and l0b_processing = True
    if l0b_processing and remove_l0a:
        campaign_dir = _get_disdrodb_directory(
            disdrodb_dir=disdrodb_dir,
            product_level="L0A",
            data_source=data_source,
            campaign_name=campaign_name,
        )
        station_product_dir = os.path.join(campaign_dir, "L0A", station_name)
        shutil.rmtree(station_product_dir)

    # ------------------------------------------------------------------------.
    # If l0b_concat=True, concat the netCDF in a single file
    if l0b_concat:
        run_disdrodb_l0b_concat_station(
            disdrodb_dir=disdrodb_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            remove_l0b=remove_l0b,
            verbose=verbose,
        )
    return None

    # -------------------------------------------------------------------------.
    # End of L0 processing for all stations
    timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
    msg = f"L0 processing of stations {station_name} completed in {timedelta_str}"
    log_info(logger, msg, verbose)
    return None


####---------------------------------------------------------------------------.
#### Wrappers to run archive L0 processing


def run_disdrodb_l0(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # L0 archive options
    l0a_processing: bool = True,
    l0b_processing: bool = True,
    l0b_concat: bool = False,
    remove_l0a: bool = False,
    remove_l0b: bool = False,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
):
    """Run the L0 processing of DISDRODB stations.

    This function enable to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    disdrodb_dir : str
        Base directory of DISDRODB
        Format: <...>/DISDRODB
    data_sources : list
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        The default is None
    campaign_names : list
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        The default is None
    station_names : list
        Station names to process.
        The default is None
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
         It takes places only if l0b_concat = True
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
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files.
        For L0B, it processes just the first 100 rows of 3 L0A files.
        The default is False.
    """
    from disdrodb.api.io import available_stations

    if l0a_processing:
        product_level = "RAW"
    elif l0b_processing:
        product_level = "L0A"
    else:
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
    if station_names is not None:
        list_info = [info for info in list_info if info[2] in station_names]
        # If nothing left, raise an error
        if len(list_info) == 0:
            raise ValueError("No stations to concatenate given the provided `station_name` argument!")

    # Print message
    n_stations = len(list_info)
    print(f"L0 processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"L0 processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
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
            # Process options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
        )
        print(f"L0 processing of {data_source} {campaign_name} {station_name} station ended.")


def run_disdrodb_l0a(
    disdrodb_dir,
    data_sources=None,
    campaign_names=None,
    station_names=None,
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
        station_names=station_names,
        # L0 archive options
        l0a_processing=True,
        l0b_processing=False,
        l0b_concat=False,
        remove_l0a=False,
        remove_l0b=False,
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
    station_names=None,
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
        station_names=station_names,
        # L0 archive options
        l0a_processing=False,
        l0b_processing=True,
        l0b_concat=False,
        remove_l0a=False,
        remove_l0b=False,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )


####--------------------------------------------------------------------------.
#### CLIck


def click_l0_station_arguments(function: object):
    """Click command line arguments for L0 processing of a station.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.argument("station_name", metavar="<station>")(function)
    function = click.argument("campaign_name", metavar="<campaign_name>")(function)
    function = click.argument("data_source", metavar="<data_source>")(function)
    function = click.argument("disdrodb_dir", metavar="<disdrodb_dir>")(function)
    return function


def click_l0_stations_options(function: object):
    """Click command line options for DISDRODB archive L0 processing.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "--data_sources",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB data sources to process",
    )(function)
    function = click.option(
        "--campaign_names",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB campaign names to process",
    )(function)
    function = click.option(
        "--station_names",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB station names to process",
    )(function)
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
    function = click.option("-v", "--verbose", type=bool, show_default=True, default=True, help="Verbose")(function)
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
        "--l0b_concat",
        type=bool,
        show_default=True,
        default=False,
        help="Produce single L0B netCDF file.",
    )(function)
    function = click.option(
        "--remove_l0b",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove all source L0B files once L0B concatenation is terminated.",
    )(function)
    function = click.option(
        "--remove_l0a",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove the L0A files once the L0B processing is terminated.",
    )(function)
    function = click.option(
        "-l0b",
        "--l0b_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Perform L0B processing.",
    )(function)
    function = click.option(
        "-l0a",
        "--l0a_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Perform L0A processing.",
    )(function)
    return function


def click_l0b_concat_options(function: object):
    """Click command line default parameters for L0B concatenation.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "--remove_l0b",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove all source L0B files once L0B concatenation is terminated.",
    )(function)
    function = click.option("-v", "--verbose", type=bool, show_default=True, default=False, help="Verbose")(function)
    return function
