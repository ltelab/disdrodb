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
from disdrodb import is_pytmatrix_available
from disdrodb.api.create_directories import (
    create_logs_directory,
    create_product_directory,
)
from disdrodb.api.info import group_filepaths
from disdrodb.api.io import find_files
from disdrodb.api.path import (
    define_accumulation_acronym,
    define_l2e_filename,
    define_l2m_filename,
)
from disdrodb.api.search import get_required_product
from disdrodb.configs import get_data_archive_dir, get_metadata_archive_dir
from disdrodb.l1.resampling import resample_dataset
from disdrodb.l2.event import get_events_info, identify_events
from disdrodb.l2.processing import (
    generate_l2_empirical,
    generate_l2_model,
    generate_l2_radar,
)
from disdrodb.l2.processing_options import get_l2_processing_options
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
from disdrodb.utils.time import ensure_sample_interval_in_seconds, get_resampling_information, regularize_dataset
from disdrodb.utils.writer import write_product

logger = logging.getLogger(__name__)


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
    l2e_options,
    # Radar options
    radar_simulation_enabled,
    radar_simulation_options,
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

        ##------------------------------------------------------------------------.
        # Remove timesteps with no drops or NaN (from L2E computations)
        # timestep_zero_drops = ds["time"].data[ds["N"].data == 0]
        # timestep_nan = ds["time"].data[np.isnan(ds["N"].data)]
        # TODO: Make it a choice !
        indices_valid_timesteps = np.where(
            ~np.logical_or(ds["N"].data == 0, np.isnan(ds["N"].data)),
        )[0]
        ds = ds.isel(time=indices_valid_timesteps)

        ##------------------------------------------------------------------------.
        #### Generate L2E product
        # TODO: Pass filtering criteria and actual L2E options !
        ds = generate_l2_empirical(ds=ds, **l2e_options)

        # Simulate L2M-based radar variables if asked
        if radar_simulation_enabled:
            ds_radar = generate_l2_radar(ds, parallel=not parallel, **radar_simulation_options)
            ds.update(ds_radar)
            ds.attrs = ds_radar.attrs.copy()

        ##------------------------------------------------------------------------.
        #### Regularize back dataset
        # TODO: infill timestep_zero_drops and timestep_nan differently ?
        # --> R, P, LWC = 0,
        # --> Z, D, with np.nan?

        ##------------------------------------------------------------------------.
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

    # -------------------------------------------------------------------------.
    # Retrieve L2 processing options
    # - Each dictionary item contains the processing options for a given rolling/accumulation_interval combo
    l2_processing_options = get_l2_processing_options()

    # ---------------------------------------------------------------------.
    # Group filepaths by sample intervals
    # - Typically the sample interval is fixed
    # - Some stations might change the sample interval along the years
    # - For each sample interval, separated processing take place here after !
    dict_filepaths = group_filepaths(filepaths, groups="sample_interval")

    # -------------------------------------------------------------------------.
    # Define list of event
    # - [(start_time, end_time)]
    # TODO: Here pass event option list !
    # TODO: Implement more general define_events function
    # - Either rainy events
    # - Either time blocks (day/month/year)
    # TODO: Define events identification settings based on accumulation
    # - This is currently done at the source sample interval !
    # - Should we allow event definition for each accumulation interval and
    #   move this code inside the loop below

    # sample_interval = list(dict_filepaths)[0]
    # filepaths = dict_filepaths[sample_interval]

    dict_list_events = {
        sample_interval: identify_events(filepaths, parallel=parallel)
        for sample_interval, filepaths in dict_filepaths.items()
    }

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
    # l2_options = l2_processing_options["1MIN"]
    available_pytmatrix = is_pytmatrix_available()

    for sample_interval_acronym, l2_options in l2_processing_options.items():

        # Retrieve accumulation_interval and rolling option
        accumulation_interval, rolling = get_resampling_information(sample_interval_acronym)

        # Retrieve radar simulation options
        radar_simulation_enabled = l2_options.get("radar_simulation_enabled", False)
        radar_simulation_options = l2_options["radar_simulation_options"]
        if not available_pytmatrix:
            radar_simulation_enabled = False

        # ------------------------------------------------------------------.
        # Group filepaths by events
        # - This is done separately for each possible source sample interval
        # - It groups filepaths by start_time and end_time provided by list_events
        # - Here 'events' can also simply be period of times ('day', 'months', ...)
        # - When aggregating/resampling/accumulating data, we need to load also
        #   some data before/after the actual event start_time/end_time
        # - get_events_info adjust the event times to accounts for the required "border" data.
        events_info = [
            get_events_info(
                list_events=list_events,
                filepaths=dict_filepaths[sample_interval],
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
                l2e_options={},  # TODO
                # Radar options
                radar_simulation_enabled=radar_simulation_enabled,
                radar_simulation_options=radar_simulation_options,
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
    l2m_options,
    # Radar options
    radar_simulation_enabled,
    radar_simulation_options,
    # Processing options
    force,
    verbose,
    parallel,  # this is used only to initialize the correct logger !
):
    # -----------------------------------------------------------------.
    # Define product name
    product = "L2M"

    # -----------------------------------------------------------------.
    # Define model options
    psd_model = l2m_options["models"][model_name]["psd_model"]
    optimization = l2m_options["models"][model_name]["optimization"]
    optimization_kwargs = l2m_options["models"][model_name]["optimization_kwargs"]
    other_options = {k: v for k, v in l2m_options.items() if k != "models"}

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
                "M1",
                "M2",
                "M3",
                "M4",
                "M5",
                "M6",
            ]
            ds = ds[variables].load()

        # Produce L2M dataset
        ds = generate_l2_model(
            ds=ds,
            psd_model=psd_model,
            optimization=optimization,
            optimization_kwargs=optimization_kwargs,
            **other_options,
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

    # -------------------------------------------------------------------------.
    # Retrieve L2 processing options
    # - Each dictionary item contains the processing options for a given rolling/accumulation_interval combo
    l2_processing_options = get_l2_processing_options()

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
    # l2_options = l2_processing_options["1MIN"]
    available_pytmatrix = is_pytmatrix_available()
    for sample_interval_acronym, l2_options in l2_processing_options.items():

        # Retrieve accumulation_interval and rolling option
        accumulation_interval, rolling = get_resampling_information(sample_interval_acronym)

        # Retrieve L2M processing options
        l2m_options = l2_options["l2m_options"]

        # Retrieve radar simulation options
        radar_simulation_enabled = l2_options.get("radar_simulation_enabled", False)
        radar_simulation_options = l2_options["radar_simulation_options"]
        if not available_pytmatrix:
            radar_simulation_enabled = False

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
        for model_name, model_options in l2m_options["models"].items():

            # Retrieve model options
            psd_model = model_options["psd_model"]
            optimization = model_options["optimization"]

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
                    l2m_options=l2m_options,
                    # Radar options
                    radar_simulation_enabled=radar_simulation_enabled,
                    radar_simulation_options=radar_simulation_options,
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
