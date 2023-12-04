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
"""Implement DISDRODB wrappers to launch L0 processing in the terminal."""

import datetime
import logging
import time

import click

from disdrodb.utils.logger import (
    # log_warning,
    # log_error,
    log_info,
)
from disdrodb.utils.scripts import _execute_cmd

logger = logging.getLogger(__name__)

####--------------------------------------------------------------------------.
#### CLIck


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


def click_remove_l0a_option(function: object):
    function = click.option(
        "--remove_l0a",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove the L0A files once the L0B processing is terminated.",
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


####--------------------------------------------------------------------------.
#### Run L0A and L0B Station processing


def run_disdrodb_l0a_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    base_dir: str = None,
):
    """Run the L0A processing of a station calling the disdrodb_l0a_station in the terminal."""
    # Define command
    cmd = " ".join([
        "disdrodb_run_l0a_station",
        # Station arguments
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
        "--base_dir",
        str(base_dir),
    ])
    # Execute command
    _execute_cmd(cmd)
    return None


def run_disdrodb_l0b_station(
    # Station arguments
    data_source,
    campaign_name,
    station_name,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    base_dir: str = None,
    remove_l0a: bool = False,
):
    """Run the L0B processing of a station calling disdrodb_run_l0b_station in the terminal."""
    # Define command
    cmd = " ".join([
        "disdrodb_run_l0b_station",
        # Station arguments
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
        "--remove_l0a",
        str(remove_l0a),
        "--base_dir",
        str(base_dir),
    ])
    # Execute command
    _execute_cmd(cmd)
    return None


def run_disdrodb_l0b_concat_station(
    data_source,
    campaign_name,
    station_name,
    remove_l0b=False,
    verbose=False,
    base_dir=None,
):
    """Concatenate the L0B files of a single DISDRODB station.

    This function runs the ``disdrodb_run_l0b_concat_station`` script in the terminal.
    """
    cmd = " ".join([
        "disdrodb_run_l0b_concat_station",
        data_source,
        campaign_name,
        station_name,
        "--remove_l0b",
        str(remove_l0b),
        "--verbose",
        str(verbose),
        "--base_dir",
        str(base_dir),
    ])
    _execute_cmd(cmd)


####--------------------------------------------------------------------------.
#### Run L0 Station processing (L0A + L0B)


def run_disdrodb_l0_station(
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
    base_dir: str = None,
):
    """Run the L0 processing of a specific DISDRODB station from the terminal.

    Parameters
    ----------
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
        If True, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with os.cpu_count().
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files for each station.
        For L0B, it processes just the first 100 rows of 3 L0A files for each station.
        The default is False.
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.
    """

    # ---------------------------------------------------------------------.
    t_i = time.time()
    msg = f"L0 processing of station {station_name} has started."
    log_info(logger=logger, msg=msg, verbose=verbose)

    # ------------------------------------------------------------------.
    # L0A processing
    if l0a_processing:
        run_disdrodb_l0a_station(
            # Station arguments
            base_dir=base_dir,
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
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Processing options
            force=force,
            verbose=verbose,
            debugging_mode=debugging_mode,
            parallel=parallel,
            remove_l0a=remove_l0a,
        )

    # ------------------------------------------------------------------------.
    # If l0b_concat=True, concat the netCDF in a single file
    if l0b_concat:
        run_disdrodb_l0b_concat_station(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            remove_l0b=remove_l0b,
            verbose=verbose,
        )

    # -------------------------------------------------------------------------.
    # End of L0 processing for all stations
    timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
    msg = f"L0 processing of stations {station_name} completed in {timedelta_str}"
    log_info(logger, msg, verbose)
    return None


####---------------------------------------------------------------------------.
#### Run L0 Archive processing


def _check_available_stations(list_info):
    # If no stations available, raise an error
    if len(list_info) == 0:
        msg = "No stations available given the provided `data_sources` and `campaign_names` arguments !"
        raise ValueError(msg)


def _filter_list_info(list_info, station_names):
    # Filter by provided stations
    if station_names is not None:
        list_info = [info for info in list_info if info[2] in station_names]
        # If nothing left, raise an error
        if len(list_info) == 0:
            raise ValueError("No stations available given the provided `station_names` argument !")
    return list_info


def _get_starting_product(l0a_processing, l0b_processing):
    if l0a_processing:
        product = "RAW"
    elif l0b_processing:
        product = "L0A"
    else:
        raise ValueError("At least l0a_processing or l0b_processing must be True.")
    return product


def run_disdrodb_l0(
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
    base_dir: str = None,
):
    """Run the L0 processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
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
        If True, the files are processed simultaneously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with os.cpu_count().
        If False, the files are processed sequentially in a single process.
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files.
        For L0B, it processes just the first 100 rows of 3 L0A files.
        The default is False.
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.
    """
    from disdrodb.api.io import available_stations

    # Get list of available stations
    product = _get_starting_product(l0a_processing=l0a_processing, l0b_processing=l0b_processing)
    list_info = available_stations(
        base_dir=base_dir,
        product=product,
        data_sources=data_sources,
        campaign_names=campaign_names,
    )
    _check_available_stations(list_info)
    list_info = _filter_list_info(list_info, station_names)

    # Print message
    n_stations = len(list_info)
    print(f"L0 processing of {n_stations} stations started.")

    # Loop over stations
    for data_source, campaign_name, station_name in list_info:
        print(f"L0 processing of {data_source} {campaign_name} {station_name} station started.")
        # Run processing
        run_disdrodb_l0_station(
            base_dir=base_dir,
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
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    base_dir: str = None,
):
    """Run the L0A processing of DISDRODB stations.

    This function allows to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
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
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is True.
    parallel : bool
        If True, the files are processed simultaneously in multiple processes.
        By default, the number of process is defined with os.cpu_count().
        If False, the files are processed sequentially in a single process.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        For L0A, it processes just the first 3 raw data files.
        The default is False.
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.
    """
    run_disdrodb_l0(
        base_dir=base_dir,
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
    data_sources=None,
    campaign_names=None,
    station_names=None,
    # Processing options
    force: bool = False,
    verbose: bool = False,
    debugging_mode: bool = False,
    parallel: bool = True,
    base_dir: str = None,
    remove_l0a: bool = False,
):
    run_disdrodb_l0(
        base_dir=base_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        # L0 archive options
        l0a_processing=False,
        l0b_processing=True,
        l0b_concat=False,
        remove_l0a=remove_l0a,
        remove_l0b=False,
        # Processing options
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        parallel=parallel,
    )


####---------------------------------------------------------------------------.
def run_disdrodb_l0b_concat(
    data_sources=None,
    campaign_names=None,
    station_names=None,
    remove_l0b=False,
    verbose=False,
    base_dir=None,
):
    """Concatenate the L0B files of the DISDRODB archive.

    This function is called by the ``disdrodb_run_l0b_concat`` script.
    """
    from disdrodb.api.io import available_stations

    list_info = available_stations(
        base_dir=base_dir,
        product="L0B",
        data_sources=data_sources,
        campaign_names=campaign_names,
    )

    _check_available_stations(list_info)
    list_info = _filter_list_info(list_info, station_names)

    # Print message
    n_stations = len(list_info)
    print(f"Concatenation of {n_stations} L0B stations started.")

    # Start the loop to launch the concatenation of each station
    for data_source, campaign_name, station_name in list_info:
        print(f"L0B files concatenation of {data_source} {campaign_name} {station_name} station started.")
        run_disdrodb_l0b_concat_station(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            remove_l0b=remove_l0b,
            verbose=verbose,
        )
    print(f"L0 files concatenation of {data_source} {campaign_name} {station_name} station ended.")


####---------------------------------------------------------------------------.
