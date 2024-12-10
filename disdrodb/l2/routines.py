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
    create_product_directory,
)
from disdrodb.api.io import get_filepaths, get_required_product
from disdrodb.api.path import (
    define_l2e_filename,
    define_l2m_filename,
    define_logs_dir,
    get_sample_interval_acronym,
)
from disdrodb.l1.resampling import (
    regularize_dataset,
    resample_dataset,
)
from disdrodb.l2.event import get_events_info, identify_events
from disdrodb.l2.processing import (
    ensure_sample_interval_in_seconds,
    generate_l2_empirical,
    generate_l2_model,
    generate_l2_radar,
)
from disdrodb.l2.processing_options import get_l2_processing_options, get_resampling_information
from disdrodb.utils.decorator import delayed_if_parallel, single_threaded_if_parallel

# Logger
from disdrodb.utils.logger import (
    close_logger,
    create_logger_file,
    define_summary_log,
    log_error,
    log_info,
)
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
    # Sampling options
    accumulation_interval,
    rolling,
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
    sample_interval_acronym = get_sample_interval_acronym(seconds=accumulation_interval)
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
    log_info(logger, msg, verbose=verbose)

    ##------------------------------------------------------------------------.
    ### Core computation
    try:
        # ------------------------------------------------------------------------.
        #### Open the dataset over the period of interest
        # - Open the netCDFs
        list_ds = [xr.open_dataset(filepath, chunks={}, cache=False, autoclose=True) for filepath in filepaths]
        # - Concatenate datasets
        ds = xr.concat(list_ds, dim="time").sel(time=slice(start_time, end_time)).compute()
        # - Close file on disk
        _ = [ds.close() for ds in list_ds]

        ##------------------------------------------------------------------------.
        #### Resample dataset
        # Here we set NaN in the raw_drop_number to 0
        # - We assume that NaN corresponds to 0
        # - When we regularize, we infill with NaN
        # - When we aggregate with sum, we don't skip NaN
        # --> Aggregation with original missing timesteps currently results in NaN !
        # TODO: tolerance on fraction of missing timesteps for large accumulation_intervals
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
        #### Generate L2E product
        ds = generate_l2_empirical(ds=ds)

        # Simulate L2M-based radar variables if asked
        if radar_simulation_enabled:
            ds_radar = generate_l2_radar(ds, parallel=not parallel, **radar_simulation_options)
            ds.update(ds_radar)
            ds.attrs = ds_radar.attrs.copy()

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
    base_dir: Optional[str] = None,
):
    """
    Generate the L2E product of a specific DISDRODB station when invoked from the terminal.

    This function is intended to be called through the ``disdrodb_run_l2e_station``
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
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    """
    # Define product
    product = "L2E"

    # ------------------------------------------------------------------------.
    # Start processing
    if verbose:
        t_i = time.time()
        msg = f"{product} processing of station {station_name} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -------------------------------------------------------------------------.
    # List L1 files to process
    required_product = get_required_product(product)
    filepaths = get_filepaths(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=required_product,
        # Processing options
        debugging_mode=False,
    )

    # -------------------------------------------------------------------------.
    # Retrieve L2 processing options
    # - Each dictionary item contains the processing options for a given rolling/accumulation_interval combo
    l2_processing_options = get_l2_processing_options()

    # -------------------------------------------------------------------------.
    # Define list event
    # TODO: pass event option list !
    list_events = identify_events(filepaths, parallel=parallel)
    if debugging_mode:
        list_events = list_events[0 : min(len(list_events), 3)]

    # ---------------------------------------------------------------------.
    # Retrieve source sample interval
    # TODO: Read from metadata ?
    with xr.open_dataset(filepaths[0], chunks={}, autoclose=True) as ds:
        sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"].data).item()

    # ---------------------------------------------------------------------.
    # Loop
    # rolling = False
    # accumulation_interval = 60
    # sample_interval_acronym = "1MIN"
    # l2_options = l2_processing_options["1MIN"]
    for sample_interval_acronym, l2_options in l2_processing_options.items():

        # Retrieve accumulation_interval and rolling option
        accumulation_interval, rolling = get_resampling_information(sample_interval_acronym)

        # Retrieve radar simulation options
        radar_simulation_enabled = l2_options.get("radar_simulation_enabled", False)
        radar_simulation_options = l2_options["radar_simulation_options"]

        # ------------------------------------------------------------------.
        # Avoid generation of rolling products for source sample interval !
        if rolling and accumulation_interval == sample_interval:
            continue

        # Avoid product generation if the accumulation_interval is less than the sample interval
        if accumulation_interval < sample_interval:
            continue

        # ------------------------------------------------------------------.
        # Create product directory
        data_dir = create_product_directory(
            base_dir=base_dir,
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
        logs_dir = define_logs_dir(
            product=product,
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Option for L2E
            sample_interval=accumulation_interval,
            rolling=rolling,
        )
        # Retrieve files events information
        events_info = get_events_info(
            list_events=list_events,
            filepaths=filepaths,
            accumulation_interval=accumulation_interval,
            rolling=rolling,
        )
        # Generate files
        # - Loop over netCDF files and generate new product files.
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
                # Sampling options
                rolling=rolling,
                accumulation_interval=accumulation_interval,
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
        define_summary_log(list_logs)

    # ---------------------------------------------------------------------.
    # End product processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
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
    distribution,
    l2m_options,
    # PSD options
    model_options,
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
    ### Core computation
    try:
        # Open the raw netCDF
        with xr.open_dataset(filepath, chunks={}, cache=False) as ds:
            ds = ds[["drop_number_concentration", "D50", "Nw", "Nt"]].load()

        # Produce L2M dataset
        ds = generate_l2_model(ds=ds, distribution=distribution, **l2m_options, **model_options)

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
                distribution=distribution,
                sample_interval=sample_interval,
                rolling=rolling,
            )
            filepath = os.path.join(data_dir, filename)
            # Write to disk
            write_product(ds, product=product, filepath=filepath, force=force)

        ##--------------------------------------------------------------------.
        # Clean environment
        del ds

        # Log end processing
        msg = f"{product} processing of {filename} has ended."
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
    base_dir: Optional[str] = None,
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
        Only the first 3 files will be processed. By default, ``False``.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    """
    # Define product
    product = "L2M"

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
    # Retrieve sampling interval
    # TODO: read from metadata !
    filepath = get_filepaths(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="L1",
        # Processing options
        debugging_mode=debugging_mode,
    )[0]
    with xr.open_dataset(filepath, chunks={}, autoclose=True) as ds:
        sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"].data).item()

    # ---------------------------------------------------------------------.
    # Loop
    # rolling = False
    # accumulation_interval = 60
    # sample_interval_acronym = "1MIN"
    # l2_options = l2_processing_options["1MIN"]
    for sample_interval_acronym, l2_options in l2_processing_options.items():

        # Retrieve accumulation_interval and rolling option
        accumulation_interval, rolling = get_resampling_information(sample_interval_acronym)

        # Retrieve L2M processing options
        l2m_options = l2_options["l2m_options"]

        # Retrieve radar simulation options
        radar_simulation_enabled = l2_options.get("radar_simulation_enabled", False)
        radar_simulation_options = l2_options["radar_simulation_options"]

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
        filepaths = get_filepaths(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=required_product,
            sample_interval=accumulation_interval,
            rolling=rolling,
            # Processing options
            debugging_mode=debugging_mode,
        )

        # -----------------------------------------------------------------.
        # Loop over distributions to fit
        # model_options =  l2_options["psd_models"]["normalized_gamma"]
        for distribution, model_options in l2_options["psd_models"].items():

            # -----------------------------------------------------------------.
            msg = f" - Fitting distribution {distribution} for sample interval {accumulation_interval} s."
            log_info(logger=logger, msg=msg, verbose=verbose)

            # -------------------------------------------------------------.
            # Create product directory
            data_dir = create_product_directory(
                base_dir=base_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product=product,
                force=force,
                # Option for L2E
                sample_interval=accumulation_interval,
                rolling=rolling,
                # Option for L2M
                distribution=distribution,
            )

            # Define logs directory
            logs_dir = define_logs_dir(
                product=product,
                base_dir=base_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                # Option for L2E
                sample_interval=accumulation_interval,
                rolling=rolling,
                # Option for L2M
                distribution=distribution,
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
                    # L2M option
                    sample_interval=accumulation_interval,
                    rolling=rolling,
                    l2m_options=l2m_options,
                    distribution=distribution,
                    model_options=model_options,
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
            define_summary_log(list_logs)

    # ---------------------------------------------------------------------.
    # End L2M processing
    if verbose:
        timedelta_str = str(datetime.timedelta(seconds=time.time() - t_i))
        msg = f"{product} processing of station {station_name} completed in {timedelta_str}"
        log_info(logger=logger, msg=msg, verbose=verbose)
