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
"""Implements routines for DISDRODB L2 processing."""

import datetime
import logging
import os
import time
from typing import Optional

import dask
import numpy as np
import pandas as pd
import xarray as xr

# Directory
from disdrodb.api.create_directories import (
    create_logs_directory,
    create_product_directory,
)
from disdrodb.api.info import get_start_end_time_from_filepaths, group_filepaths
from disdrodb.api.io import find_files
from disdrodb.api.path import (
    define_accumulation_acronym,
    define_l2e_filename,
    define_l2m_filename,
)
from disdrodb.api.search import get_required_product
from disdrodb.configs import (
    get_data_archive_dir,
    get_metadata_archive_dir,
    get_model_options,
    get_product_options,
    get_product_time_integrations,
)
from disdrodb.l1.resampling import resample_dataset
from disdrodb.l2.event import get_events_info, group_timesteps_into_event
from disdrodb.l2.processing import (
    generate_l2_empirical,
    generate_l2_model,
    generate_l2_radar,
)
from disdrodb.metadata import read_station_metadata
from disdrodb.utils.decorators import delayed_if_parallel, single_threaded_if_parallel
from disdrodb.utils.list import flatten_list

# Logger
from disdrodb.utils.logger import (
    close_logger,
    create_logger_file,
    create_product_logs,
    log_error,
    log_info,
)
from disdrodb.utils.time import (
    ensure_sample_interval_in_seconds,
    ensure_sorted_by_time,
    generate_time_blocks,
    get_resampling_information,
    regularize_dataset,
)
from disdrodb.utils.writer import write_product

logger = logging.getLogger(__name__)


@dask.delayed
def _delayed_open_dataset(filepath):
    with dask.config.set(scheduler="synchronous"):
        ds = xr.open_dataset(filepath, chunks={}, autoclose=True, decode_timedelta=False, cache=False)
    return ds


####----------------------------------------------------------------------------.
def identify_events(
    filepaths,
    parallel=False,
    min_n_drops=5,
    neighbor_min_size=2,
    neighbor_time_interval="5MIN",
    intra_event_max_time_gap="6H",
    event_min_duration="5MIN",
    event_min_size=3,
):
    """Return a list of rainy events.

    Rainy timesteps are defined when N > min_n_drops.
    Any rainy isolated timesteps (based on neighborhood criteria) is removed.
    Then, consecutive rainy timesteps are grouped into the same event if the time gap between them does not
    exceed `intra_event_max_time_gap`. Finally, events that do not meet minimum size or duration
    requirements are filtered out.

    Parameters
    ----------
    filepaths: list
        List of L1C file paths.
    parallel: bool
        Whether to load the files in parallel.
        Set parallel=True only in a multiprocessing environment.
        The default is False.
    neighbor_time_interval : str
        The time interval around a given a timestep defining the neighborhood.
        Only timesteps that fall within this time interval before or after a timestep are considered neighbors.
    neighbor_min_size : int, optional
        The minimum number of neighboring timesteps required within `neighbor_time_interval` for a
        timestep to be considered non-isolated.  Isolated timesteps are removed !
        - If `neighbor_min_size=0,  then no timestep is considered isolated and no filtering occurs.
        - If `neighbor_min_size=1`, the timestep must have at least one neighbor within `neighbor_time_interval`.
        - If `neighbor_min_size=2`, the timestep must have at least two timesteps within `neighbor_time_interval`.
        Defaults to 1.
    intra_event_max_time_gap: str
        The maximum time interval between two timesteps to be considered part of the same event.
        This parameters is used to group timesteps into events !
    event_min_duration : str
        The minimum duration an event must span. Events shorter than this duration are discarded.
    event_min_size : int, optional
        The minimum number of valid timesteps required for an event. Defaults to 1.

    Returns
    -------
    list of dict
        A list of events, where each event is represented as a dictionary with keys:
        - "start_time": np.datetime64, start time of the event
        - "end_time": np.datetime64, end time of the event
        - "duration": np.timedelta64, duration of the event
        - "n_timesteps": int, number of valid timesteps in the event
    """
    # Open datasets in parallel
    if parallel:
        list_ds = dask.compute([_delayed_open_dataset(filepath) for filepath in filepaths])[0]
    else:
        list_ds = [xr.open_dataset(filepath, chunks={}, cache=False, decode_timedelta=False) for filepath in filepaths]
    # Filter dataset for requested variables
    variables = ["time", "N"]
    list_ds = [ds[variables] for ds in list_ds]
    # Concat datasets
    ds = xr.concat(list_ds, dim="time", compat="no_conflicts", combine_attrs="override")
    # Read in memory the variable needed
    ds = ds.compute()
    # Close file on disk
    _ = [ds.close() for ds in list_ds]
    del list_ds
    # Sort dataset by time
    ds = ensure_sorted_by_time(ds)
    # Define candidate timesteps to group into events
    idx_valid = ds["N"].data > min_n_drops
    timesteps = ds["time"].data[idx_valid]
    # Define event list
    event_list = group_timesteps_into_event(
        timesteps=timesteps,
        neighbor_min_size=neighbor_min_size,
        neighbor_time_interval=neighbor_time_interval,
        intra_event_max_time_gap=intra_event_max_time_gap,
        event_min_duration=event_min_duration,
        event_min_size=event_min_size,
    )
    return event_list


def identify_time_partitions(filepaths: list[str], freq: str) -> list[dict]:
    """Identify the set of time blocks covered by files.

    The result is a minimal, sorted, and unique set of time partitions.

    Parameters
    ----------
    filepaths : list of str
        Paths to input files from which start and end times will be extracted
        via `get_start_end_time_from_filepaths`.
    freq : {'none', 'hour', 'day', 'month', 'quarter', 'season', 'year'}
        Frequency determining the granularity of candidate blocks.
        See `generate_time_blocks` for more details.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing:

        - `start_time` (numpy.datetime64[s])
            Inclusive start of a time block.
        - `end_time` (numpy.datetime64[s])
            Inclusive end of a time block.

        Only those blocks that overlap at least one file's interval are returned.
        The list is sorted by `start_time` and contains no duplicate blocks.
    """
    # Define file start time and end time
    start_times, end_times = get_start_end_time_from_filepaths(filepaths)

    # Define files time coverage
    start_time, end_time = start_times.min(), end_times.max()

    # Compute candidate time blocks
    blocks = generate_time_blocks(start_time, end_time, freq=freq)  # end_time non inclusive is correct?

    # Select time blocks with files
    mask = (blocks[:, 0][:, None] <= end_times) & (blocks[:, 1][:, None] >= start_times)
    blocks = blocks[mask.any(axis=1)]

    # Ensure sorted unique time blocks
    order = np.argsort(blocks[:, 0])
    blocks = np.unique(blocks[order], axis=0)

    # Convert to list of dicts
    list_time_blocks = [{"start_time": start_time, "end_time": end_time} for start_time, end_time in blocks]
    return list_time_blocks


def is_possible_product(accumulation_interval, sample_interval, rolling):
    """Assess if production is possible given the requested accumulation interval and source sample_interval."""
    # Avoid rolling product generation at source sample interval
    if rolling and accumulation_interval == sample_interval:
        return False
    # Avoid product generation if the accumulation_interval is less than the sample interval
    if accumulation_interval < sample_interval:
        return False
    # Avoid producti generation if accumulation_interval is not multiple of sample_interval
    return accumulation_interval % sample_interval == 0


def get_list_events(filepaths, parallel, l2_processing_options):
    # - [(start_time, end_time)]
    # - Either save_by_event or save_by_time_block
    # - save_by_event requires loading data into memory
    #   --> Does some data selection on what to process !
    # - save_by_time_block does not require loading data into memory
    #   --> Does not do some data selection on what to process !

    # save_by_event
    # - event options

    # save_by_time_block
    # - year, season, month, day

    # TODO: Here pass event option list !
    #  - min_n_drops=5,
    #  - neighbor_min_size=2,
    #  - neighbor_time_interval="5MIN",
    #  - intra_event_max_time_gap="6H",
    #  - event_min_duration="5MIN",
    #  - event_min_size=3,

    freq = "month"

    # l2_processing_options
    save_by_event = False

    if save_by_event:
        return identify_events(filepaths, parallel=parallel)

    return identify_time_partitions(filepaths, freq=freq)


####----------------------------------------------------------------------------.
#### L2E


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l2e(
    start_time,
    end_time,
    filepaths,
    data_dir,
    logs_dir,
    campaign_name,
    station_name,
    # L2E options
    accumulation_interval,
    rolling,
    product_options,
    # Processing options
    force,
    verbose,
    parallel,  # this is used by the decorator and to initialize correctly the logger !
):
    # -----------------------------------------------------------------.
    # Define product name
    product = "L2E"

    # -----------------------------------------------------------------.
    # Create file logger
    sample_interval_acronym = define_accumulation_acronym(seconds=accumulation_interval, rolling=rolling)
    starting_time = pd.to_datetime(start_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(end_time).strftime("%Y%m%d%H%M%S")
    filename = f"L2E.{sample_interval_acronym}.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}"
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
    ### Core computation
    try:
        # ------------------------------------------------------------------------.
        #### Open the dataset over the period of interest
        # - Open the netCDFs
        list_ds = [
            xr.open_dataset(filepath, chunks={}, decode_timedelta=False, cache=False, autoclose=True)
            for filepath in filepaths
        ]
        # - Concatenate datasets
        ds = xr.concat(list_ds, dim="time", compat="no_conflicts", combine_attrs="override")
        ds = ds.sel(time=slice(start_time, end_time)).compute()
        # - Close file on disk
        _ = [ds.close() for ds in list_ds]

        ##------------------------------------------------------------------------.
        #### Resample dataset
        # Here we set NaN in the raw_drop_number to 0
        # - We assume that NaN corresponds to 0
        # - When we regularize, we infill with NaN
        # - When we aggregate with sum, we don't skip NaN
        # --> Aggregation with original missing timesteps currently results in NaN !
        # TODO: Add tolerance on fraction of missing timesteps for large accumulation_intervals
        # TODO: NaN should not be set as 0 !

        ds["raw_drop_number"] = xr.where(np.isnan(ds["raw_drop_number"]), 0, ds["raw_drop_number"])
        ds["drop_number"] = xr.where(np.isnan(ds["drop_number"]), 0, ds["drop_number"])

        # - Regularize dataset
        # --> Infill missing timesteps with np.Nan
        sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"]).item()
        ds = regularize_dataset(ds, freq=f"{sample_interval}s")

        # - Resample dataset
        ds = resample_dataset(
            ds=ds,
            sample_interval=sample_interval,
            accumulation_interval=accumulation_interval,
            rolling=rolling,
        )

        # Extract L2E processing options
        drop_timesteps_without_drops = product_options.pop("drop_timesteps_without_drops")
        radar_simulation_enabled = product_options.pop("radar_simulation_enabled")
        radar_simulation_options = product_options.pop("radar_simulation_options")

        ##------------------------------------------------------------------------.
        #### Preprocessing
        # Optionally remove timesteps with no drops or NaN
        # --> More aggressive filtering can be done at the time of analysis
        # --> To not discard data:
        #     - drop_timesteps_without_drops=False
        #     - Or a posteriori use regularize_dataset (but info missing timesteps lost)
        if drop_timesteps_without_drops:
            indices_valid_timesteps = np.where(
                ~np.logical_or(ds["N"].data == 0, np.isnan(ds["N"].data)),
            )[0]
            ds = ds.isel(time=indices_valid_timesteps)

        ##------------------------------------------------------------------------.
        #### Generate L2E product
        # - Only if at least 2 timesteps available
        if len(ds["time"].size > 2):

            # Compute L2E variables
            ds = generate_l2_empirical(ds=ds, **product_options)

            # Simulate L2M-based radar variables if asked
            if radar_simulation_enabled:
                ds_radar = generate_l2_radar(ds, parallel=not parallel, **radar_simulation_options)
                ds.update(ds_radar)
                ds.attrs = ds_radar.attrs.copy()

            # Write netCDF4 dataset
            if ds["time"].size > 1:
                filename = define_l2e_filename(
                    ds,
                    campaign_name=campaign_name,
                    station_name=station_name,
                    sample_interval=accumulation_interval,
                    rolling=rolling,
                )
                filepath = os.path.join(data_dir, filename)
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


def run_l2e_station(
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
    Generate the L2E product of a specific DISDRODB station when invoked from the terminal.

    This function is intended to be called through the ``disdrodb_run_l2e_station``
    command-line interface.

    The DISDRODB L2E routine generate a L2E file for each event.
    Events are defined based on the DISDRODB event settings options.
    The DISDRODB event settings allows to produce L2E files either
    per custom block of time (i.e day/month/year) or for blocks of rainy events.

    For stations with varying measurement intervals, DISDRODB defines a separate list of 'events'
    for each measurement interval option. In other words, DISDRODB does not
    mix files with data acquired at different sample intervals when resampling the data.

    L0C product generation ensure creation of files with unique sample intervals.

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
    product = "L2E"

    # Define base directory
    data_archive_dir = get_data_archive_dir(data_archive_dir)

    # Retrieve DISDRODB Metadata Archive directory
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir=metadata_archive_dir)

    # ------------------------------------------------------------------------.
    # Start processing
    if verbose:
        t_i = time.time()
        msg = f"{product} processing of station {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -------------------------------------------------------------------------.
    # List L1 files to process
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
            debugging_mode=False,
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

    # ---------------------------------------------------------------------.
    # Group filepaths by source sample intervals
    # - Typically the sample interval is fixed and is just one
    # - Some stations might change the sample interval along the years
    # - For each sample interval, separated processing take place here after !
    dict_filepaths = group_filepaths(filepaths, groups="sample_interval")

    # -------------------------------------------------------------------------.
    # Define list of "events"
    # - [{start_time:xxx, end_time: xxx}, ....]
    # - Either save_by_event or save_by_time_block
    # - save_by_event requires loading data into memory
    #   --> Does some data selection on what to process !
    # - save_by_time_block does not require loading data into memory
    #   --> Does not do some data selection on what to process !

    # [IMPROVEMENT] allow custom event definition for each accumulation interval?
    l2_processing_options = get_product_options("L2E")
    dict_list_events = {
        sample_interval: get_list_events(filepaths, parallel=parallel, l2_processing_options=l2_processing_options)
        for sample_interval, filepaths in dict_filepaths.items()
    }

    # sample_interval = list(dict_filepaths)[0]
    # filepaths = dict_filepaths[sample_interval]

    # ---------------------------------------------------------------------.
    # Subset for debugging mode
    if debugging_mode:
        dict_list_events = {
            sample_interval: list_events[0 : min(len(list_events), 3)]
            for sample_interval, list_events in dict_list_events.items()
        }

    # ---------------------------------------------------------------------.
    # Loop
    # rolling = False
    # accumulation_interval = 60
    # sample_interval_acronym = "1MIN"
    # product_options = l2_processing_options["1MIN"]
    time_integrations = get_product_time_integrations("L2E")
    for sample_interval_acronym in time_integrations:

        # Retrieve product options
        product_options = get_product_options("L2E", time_integration=sample_interval_acronym)

        # Retrieve accumulation_interval and rolling option
        accumulation_interval, rolling = get_resampling_information(sample_interval_acronym)

        # ------------------------------------------------------------------.
        # Group filepaths by events
        # - This is done separately for each possible source sample interval
        # - It groups filepaths by start_time and end_time provided by list_events
        # - Here 'events' can also simply be period of times ('day', 'months', ...)
        # - When aggregating/resampling/accumulating data, we need to load also
        #   some data after the actual event end_time to ensure that the resampled dataset
        #   contains the event_end_time
        #   --> get_events_info adjust the event end_time to accounts for the required "border" data.
        events_info = [
            get_events_info(
                list_events=list_events,
                filepaths=dict_filepaths[sample_interval],
                sample_interval=sample_interval,
                accumulation_interval=accumulation_interval,
                rolling=rolling,
            )
            for sample_interval, list_events in dict_list_events.items()
            if is_possible_product(
                accumulation_interval=accumulation_interval,
                sample_interval=sample_interval,
                rolling=rolling,
            )
        ]
        events_info = flatten_list(events_info)

        # ------------------------------------------------------------------.
        # Skip processing if no files available
        # - When not compatible accumulation_interval with source sample_interval
        if len(events_info) == 0:
            continue

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
            # Option for L2E
            sample_interval=accumulation_interval,
            rolling=rolling,
        )

        # Define logs directory
        logs_dir = create_logs_directory(
            product=product,
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Option for L2E
            sample_interval=accumulation_interval,
            rolling=rolling,
        )

        # ------------------------------------------------------------------.
        # Generate files
        # - L2E product generation is optionally parallelized over events
        # - If parallel=True, it does that in parallel using dask.delayed
        list_tasks = [
            _generate_l2e(
                start_time=event_info["start_time"],
                end_time=event_info["end_time"],
                filepaths=event_info["filepaths"],
                data_dir=data_dir,
                logs_dir=logs_dir,
                campaign_name=campaign_name,
                station_name=station_name,
                # L2E options
                rolling=rolling,
                accumulation_interval=accumulation_interval,
                product_options=product_options,
                # Processing options
                force=force,
                verbose=verbose,
                parallel=parallel,
            )
            for event_info in events_info
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
            # Product options
            sample_interval=accumulation_interval,
            rolling=rolling,
            # Logs list
            list_logs=list_logs,
        )

    # ---------------------------------------------------------------------.
    # End product processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
        msg = f"{product} processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)


####----------------------------------------------------------------------------.
#### L2M


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l2m(
    filepath,
    data_dir,
    logs_dir,
    campaign_name,
    station_name,
    # L2M options
    sample_interval,
    rolling,
    model_name,
    product_options,
    # Processing options
    force,
    verbose,
    parallel,  # this is used only to initialize the correct logger !
):
    # -----------------------------------------------------------------.
    # Define product name
    product = "L2M"

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

    # Define variables to load
    optimization_kwargs = product_options["optimization_kwargs"]
    if "init_method" in optimization_kwargs:
        init_method = optimization_kwargs["init_method"]
        moments = [f"M{order}" for order in init_method.replace("M", "")] + ["M1"]
    else:
        moments = ["M1"]

    ##------------------------------------------------------------------------.
    # Extract L2M processing options
    radar_simulation_enabled = product_options.pop("radar_simulation_enabled")
    radar_simulation_options = product_options.pop("radar_simulation_options")

    ##------------------------------------------------------------------------
    ### Core computation
    try:
        # Open the raw netCDF
        with xr.open_dataset(filepath, chunks={}, decode_timedelta=False, cache=False) as ds:
            variables = [
                "drop_number_concentration",
                "fall_velocity",
                "D50",
                "Nw",
                "Nt",
                *moments,
            ]
            ds = ds[variables].load()

        # Produce L2M dataset
        ds = generate_l2_model(
            ds=ds,
            **product_options,
        )

        # Simulate L2M-based radar variables if asked
        if radar_simulation_enabled:
            ds_radar = generate_l2_radar(ds, parallel=not parallel, **radar_simulation_options)
            ds.update(ds_radar)
            ds.attrs = ds_radar.attrs.copy()

        # Write L2M netCDF4 dataset
        if ds["time"].size > 1:
            # Define filepath
            filename = define_l2m_filename(
                ds,
                campaign_name=campaign_name,
                station_name=station_name,
                sample_interval=sample_interval,
                rolling=rolling,
                model_name=model_name,
            )
            filepath = os.path.join(data_dir, filename)
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


def run_l2m_station(
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
    Run the L2M processing of a specific DISDRODB station when invoked from the terminal.

    This function is intended to be called through the ``disdrodb_run_l2m_station``
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
    product = "L2M"

    # Define base directory
    data_archive_dir = get_data_archive_dir(data_archive_dir)

    # Retrieve DISDRODB Metadata Archive directory
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)

    # ------------------------------------------------------------------------.
    # Start processing
    if verbose:
        t_i = time.time()
        msg = f"{product} processing of station {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # ---------------------------------------------------------------------.
    # Retrieve source sampling interval
    # - If a station has varying measurement interval over time, choose the smallest one !
    metadata = read_station_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    sample_interval = metadata["measurement_interval"]
    if isinstance(sample_interval, list):
        sample_interval = min(sample_interval)

    # ---------------------------------------------------------------------.
    # Loop
    # sample_interval_acronym = "1MIN"
    time_integrations = get_product_time_integrations("L2M")
    for sample_interval_acronym in time_integrations:

        # Retrieve product options
        product_options = get_product_options("L2M", time_integration=sample_interval_acronym)

        # Retrieve accumulation_interval and rolling option
        accumulation_interval, rolling = get_resampling_information(sample_interval_acronym)

        # ------------------------------------------------------------------.
        # Avoid generation of rolling products for source sample interval !
        if rolling and accumulation_interval == sample_interval:
            continue

        # Avoid product generation if the accumulation_interval is less than the sample interval
        if accumulation_interval < sample_interval:
            continue

        # -----------------------------------------------------------------.
        # List files to process
        required_product = get_required_product(product)
        flag_not_available_data = False
        try:
            filepaths = find_files(
                data_archive_dir=data_archive_dir,
                # Station arguments
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                # Product options
                product=required_product,
                sample_interval=accumulation_interval,
                rolling=rolling,
                # Processing options
                debugging_mode=debugging_mode,
            )
        except Exception as e:
            print(str(e))  # Case where no file paths available
            flag_not_available_data = True

        # If no data available, try with other L2E accumulation intervals
        if flag_not_available_data:
            msg = (
                f"{product} processing of {data_source} {campaign_name} {station_name}"
                + f"has not been launched because of missing {required_product} {sample_interval_acronym} data ."
            )
            print(msg)
            continue

        # -----------------------------------------------------------------.
        # Loop over distributions to fit
        # model_name = "GAMMA_ML"
        # model_options =  l2m_options["models"][model_name]
        # Retrieve list of models to fit
        models = product_options.pop("models")
        for model_name in models:

            # Retrieve model options
            model_options = get_model_options(product="L2M", model_name=model_name)
            psd_model = model_options["psd_model"]
            optimization = model_options["optimization"]
            product_options.update(model_options)

            # -----------------------------------------------------------------.
            msg = f"Production of L2M_{model_name} for sample interval {accumulation_interval} s has started."
            log_info(logger=logger, msg=msg, verbose=verbose)
            msg = f"Estimating {psd_model} parameters using {optimization}."
            log_info(logger=logger, msg=msg, verbose=verbose)

            # -------------------------------------------------------------.
            # Create product directory
            data_dir = create_product_directory(
                # DISDRODB root directories
                data_archive_dir=data_archive_dir,
                metadata_archive_dir=metadata_archive_dir,
                # Station arguments
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                # Processing options
                product=product,
                force=force,
                # Option for L2E
                sample_interval=accumulation_interval,
                rolling=rolling,
                # Option for L2M
                model_name=model_name,
            )

            # Define logs directory
            logs_dir = create_logs_directory(
                product=product,
                data_archive_dir=data_archive_dir,
                # Station arguments
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                # Option for L2E
                sample_interval=accumulation_interval,
                rolling=rolling,
                # Option for L2M
                model_name=model_name,
            )

            # Generate L2M files
            # - Loop over the L2E netCDF files and generate L2M files.
            # - If parallel=True, it does that in parallel using dask.delayed
            list_tasks = [
                _generate_l2m(
                    filepath=filepath,
                    data_dir=data_dir,
                    logs_dir=logs_dir,
                    campaign_name=campaign_name,
                    station_name=station_name,
                    # L2M options
                    sample_interval=accumulation_interval,
                    rolling=rolling,
                    model_name=model_name,
                    product_options=product_options,
                    # Processing options
                    force=force,
                    verbose=verbose,
                    parallel=parallel,
                )
                for filepath in filepaths
            ]
            list_logs = dask.compute(*list_tasks) if parallel else list_tasks

            # -----------------------------------------------------------------.
            # Define L2M summary logs
            create_product_logs(
                product=product,
                # Station arguments
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                # DISDRODB root directory
                data_archive_dir=data_archive_dir,
                # Product options
                model_name=model_name,
                sample_interval=sample_interval,
                rolling=rolling,
                # Logs list
                list_logs=list_logs,
            )

    # ---------------------------------------------------------------------.
    # End L2M processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=round(time.time() - t_i)))
        msg = f"{product} processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)
