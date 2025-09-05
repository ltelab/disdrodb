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

import copy
import datetime
import json
import logging
import os
import time
from typing import Optional

import pandas as pd

from disdrodb.api.checks import check_station_inputs
from disdrodb.api.create_directories import (
    create_logs_directory,
    create_product_directory,
)
from disdrodb.api.info import group_filepaths
from disdrodb.api.io import open_netcdf_files
from disdrodb.api.path import (
    define_file_folder_path,
    define_l2e_filename,
    define_l2m_filename,
    define_temporal_resolution,
)
from disdrodb.api.search import get_required_product
from disdrodb.configs import (
    get_data_archive_dir,
    get_metadata_archive_dir,
    get_model_options,
    get_product_options,
    get_product_temporal_resolutions,
)
from disdrodb.l1.resampling import resample_dataset
from disdrodb.l2.processing import (
    generate_l2_radar,
    generate_l2e,
    generate_l2m,
)
from disdrodb.metadata import read_station_metadata
from disdrodb.scattering.routines import precompute_scattering_tables
from disdrodb.utils.archiving import define_temporal_partitions, get_files_partitions
from disdrodb.utils.dask import execute_tasks_safely
from disdrodb.utils.decorators import delayed_if_parallel, single_threaded_if_parallel
from disdrodb.utils.list import flatten_list

# Logger
from disdrodb.utils.logger import (
    create_product_logs,
    log_info,
)
from disdrodb.utils.routines import (
    is_possible_product,
    run_product_generation,
    try_get_required_filepaths,
)
from disdrodb.utils.time import (
    ensure_sample_interval_in_seconds,
    get_resampling_information,
)
from disdrodb.utils.writer import write_product

logger = logging.getLogger(__name__)


####----------------------------------------------------------------------------.


class ProcessingOptions:
    """Define L2 products processing options."""

    # TODO: TO MOVE ELSEWHERE (AFTER L1 REFACTORING !)

    def __init__(self, product, filepaths, parallel, temporal_resolutions=None):
        """Define L2 products processing options."""
        import disdrodb

        # ---------------------------------------------------------------------.
        # Define temporal resolutions for which to retrieve processing options
        if temporal_resolutions is None:
            temporal_resolutions = get_product_temporal_resolutions(product)
        elif isinstance(temporal_resolutions, str):
            temporal_resolutions = [temporal_resolutions]

        # ---------------------------------------------------------------------.
        # Get product options at various temporal resolutions
        dict_product_options = {
            temporal_resolution: get_product_options(product, temporal_resolution=temporal_resolution)
            for temporal_resolution in temporal_resolutions
        }

        # ---------------------------------------------------------------------.
        # Group filepaths by source sample intervals
        # - Typically the sample interval is fixed and is just one
        # - Some stations might change the sample interval along the years
        # - For each sample interval, separated processing take place here after !
        dict_filepaths = group_filepaths(filepaths, groups="sample_interval")

        # ---------------------------------------------------------------------.
        # Retrieve processing information for each temporal resolution
        dict_folder_partitioning = {}
        dict_files_partitions = {}
        _cache_dict_list_partitions: dict[str, dict] = {}
        for temporal_resolution in temporal_resolutions:

            # -------------------------------------------------------------------------.
            # Retrieve product options
            product_options = dict_product_options[temporal_resolution].copy()

            # Retrieve accumulation_interval and rolling option
            accumulation_interval, rolling = get_resampling_information(temporal_resolution)

            # Extract processing options
            archive_options = product_options.pop("archive_options")

            dict_product_options[temporal_resolution] = product_options
            # -------------------------------------------------------------------------.
            # Define folder partitioning
            if "folder_partitioning" not in archive_options:
                dict_folder_partitioning[temporal_resolution] = disdrodb.config.get("folder_partitioning")
            else:
                dict_folder_partitioning[temporal_resolution] = archive_options.pop("folder_partitioning")

            # -------------------------------------------------------------------------.
            # Define list of temporal partitions
            # - [{start_time: np.datetime64, end_time: np.datetime64}, ....]
            # - Either strategy: "event" or "time_block" or save_by_time_block"
            # - "event" requires loading data into memory to identify events
            #   --> Does some data filtering on what to process !
            # - "time_block" does not require loading data into memory
            #   --> Does not do data filtering on what to process !
            # --> Here we cache dict_list_partitions so that we don't need to recompute
            #     stuffs if processing options are the same
            key = json.dumps(archive_options, sort_keys=True)
            if key not in _cache_dict_list_partitions:
                _cache_dict_list_partitions[key] = {
                    sample_interval: define_temporal_partitions(filepaths, parallel=parallel, **archive_options)
                    for sample_interval, filepaths in dict_filepaths.items()
                }
            dict_list_partitions = _cache_dict_list_partitions[key].copy()  # To avoid in-place replacement

            # ------------------------------------------------------------------.
            # Group filepaths by temporal partitions
            # - This is done separately for each possible source sample interval
            # - It groups filepaths by start_time and end_time provided by list_partitions
            # - Here 'events' can also simply be period of times ('day', 'months', ...)
            # - When aggregating/resampling/accumulating data, we need to load also
            #   some data after the actual event end_time to ensure that the resampled dataset
            #   contains the event_end_time
            #   --> get_files_partitions adjust the event end_time to accounts for the required "border" data.
            # - ATTENTION: get_files_partitions returns start_time and end_time as datetime objects !
            files_partitions = [
                get_files_partitions(
                    list_partitions=list_partitions,
                    filepaths=dict_filepaths[sample_interval],
                    sample_interval=sample_interval,
                    accumulation_interval=accumulation_interval,
                    rolling=rolling,
                )
                for sample_interval, list_partitions in dict_list_partitions.items()
                if product != "L2E"
                or is_possible_product(
                    accumulation_interval=accumulation_interval,
                    sample_interval=sample_interval,
                    rolling=rolling,
                )
            ]
            files_partitions = flatten_list(files_partitions)
            dict_files_partitions[temporal_resolution] = files_partitions

        # ------------------------------------------------------------------.
        # Keep only temporal_resolutions for which events could be defined
        # - Remove e.g when not compatible accumulation_interval with source sample_interval
        temporal_resolutions = [
            temporal_resolution
            for temporal_resolution in temporal_resolutions
            if len(dict_files_partitions[temporal_resolution]) > 0
        ]
        # ------------------------------------------------------------------.
        # Add attributes
        self.temporal_resolutions = temporal_resolutions
        self.dict_files_partitions = dict_files_partitions
        self.dict_product_options = dict_product_options
        self.dict_folder_partitioning = dict_folder_partitioning

    def get_files_partitions(self, temporal_resolution):
        """Return files partitions dictionary for a specific L2E product."""
        return self.dict_files_partitions[temporal_resolution]

    def get_product_options(self, temporal_resolution):
        """Return product options dictionary for a specific L2E product."""
        return self.dict_product_options[temporal_resolution]

    def get_folder_partitioning(self, temporal_resolution):
        """Return the folder partitioning for a specific L2E product."""
        # to be used for logs and files !
        return self.dict_folder_partitioning[temporal_resolution]


####----------------------------------------------------------------------------.
#### L2E


def define_l2e_logs_filename(campaign_name, station_name, start_time, end_time, accumulation_interval, rolling):
    """Define L2E logs filename."""
    temporal_resolution = define_temporal_resolution(seconds=accumulation_interval, rolling=rolling)
    starting_time = pd.to_datetime(start_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(end_time).strftime("%Y%m%d%H%M%S")
    logs_filename = f"L2E.{temporal_resolution}.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}"
    return logs_filename


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l2e(
    start_time,
    end_time,
    filepaths,
    data_dir,
    logs_dir,
    logs_filename,
    folder_partitioning,
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
    """Generate the L2E product from the DISDRODB L1 netCDF file."""
    # Define product
    product = "L2E"

    # Define product processing function
    def core(
        filepaths,
        campaign_name,
        station_name,
        product_options,
        # Processing options
        logger,
        parallel,
        verbose,
        force,
        # Resampling arguments
        start_time,
        end_time,
        accumulation_interval,
        rolling,
        # Archiving arguments
        data_dir,
        folder_partitioning,
    ):
        """Define L1 product processing."""
        # Copy to avoid in-place replacement (outside this function)
        product_options = product_options.copy()

        # Open the dataset over the period of interest
        ds = open_netcdf_files(filepaths, start_time=start_time, end_time=end_time, parallel=False)
        ds = ds.load()
        ds.close()

        # Resample dataset # TODO: in future to perform in L1
        # - Define sample interval in seconds
        sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"]).to_numpy().item()
        # - Resample dataset
        ds = resample_dataset(
            ds=ds,
            sample_interval=sample_interval,
            accumulation_interval=accumulation_interval,
            rolling=rolling,
        )

        # Extract L2E processing options
        l2e_options = product_options.get("product_options")
        radar_enabled = product_options.get("radar_enabled")
        radar_options = product_options.get("radar_options")

        # Ensure at least 2 timestep available
        if ds["time"].size < 2:
            log_info(logger=logger, msg="File not created. Less than two timesteps available.", verbose=verbose)
            return None

        # Compute L2E variables
        ds = generate_l2e(ds=ds, **l2e_options)

        # Ensure at least 2 timestep available
        if ds["time"].size < 2:
            log_info(logger=logger, msg="File not created. Less than two timesteps available.", verbose=verbose)
            return None

        # Simulate L2M-based radar variables if asked
        if radar_enabled:
            ds_radar = generate_l2_radar(ds, parallel=not parallel, **radar_options)
            ds.update(ds_radar)
            ds.attrs = ds_radar.attrs.copy()

        # Write L2E netCDF4 dataset
        filename = define_l2e_filename(
            ds,
            campaign_name=campaign_name,
            station_name=station_name,
            sample_interval=accumulation_interval,
            rolling=rolling,
        )
        folder_path = define_file_folder_path(ds, dir_path=data_dir, folder_partitioning=folder_partitioning)
        filepath = os.path.join(folder_path, filename)
        write_product(ds, filepath=filepath, force=force)

        # Return L2E dataset
        return ds

    # Define product processing function kwargs
    core_func_kwargs = dict(  # noqa: C408
        filepaths=filepaths,
        campaign_name=campaign_name,
        station_name=station_name,
        product_options=product_options,
        # Resampling arguments
        start_time=start_time,
        end_time=end_time,
        accumulation_interval=accumulation_interval,
        rolling=rolling,
        # Archiving arguments
        data_dir=data_dir,
        folder_partitioning=folder_partitioning,
        # Processing options
        parallel=parallel,
        verbose=verbose,
        force=force,
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

    This routine generates L2E files.
    Files are defined based on the DISDRODB archive settings options.
    The DISDRODB archive settings allows to produce L2E files either
    per custom block of time (i.e day/month/year) or per blocks of (rainy) events.

    For stations with varying measurement intervals, DISDRODB defines a separate list of partitions
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

    # Check valid data_source, campaign_name, and station_name
    check_station_inputs(
        metadata_archive_dir=metadata_archive_dir,
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

    # -------------------------------------------------------------------------.
    # List files to process
    # - If no data available, print error message and return None
    required_product = get_required_product(product)
    filepaths = try_get_required_filepaths(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=required_product,
        # Processing options
        debugging_mode=debugging_mode,
    )
    if filepaths is None:
        return

    # -------------------------------------------------------------------------.
    # Retrieve L2E processing options
    l2e_processing_options = ProcessingOptions(product="L2E", filepaths=filepaths, parallel=parallel)

    # -------------------------------------------------------------------------.
    # Generate products for each temporal resolution
    # rolling = False
    # accumulation_interval = 60
    # temporal_resolution = "10MIN"
    # folder_partitioning = ""
    # product_options = l2e_processing_options.get_product_options(temporal_resolution)

    for temporal_resolution in l2e_processing_options.temporal_resolutions:
        # Print progress message
        msg = f"Production of {product} {temporal_resolution} has started."
        log_info(logger=logger, msg=msg, verbose=verbose)

        # Retrieve event info
        files_partitions = l2e_processing_options.get_files_partitions(temporal_resolution)

        # Retrieve folder partitioning (for files and logs)
        folder_partitioning = l2e_processing_options.get_folder_partitioning(temporal_resolution)

        # Retrieve product options
        product_options = l2e_processing_options.get_product_options(temporal_resolution)

        # Retrieve accumulation_interval and rolling option
        accumulation_interval, rolling = get_resampling_information(temporal_resolution)

        # Precompute required scattering tables
        if product_options["radar_enabled"]:
            radar_options = product_options["radar_options"]
            precompute_scattering_tables(verbose=verbose, **radar_options)

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
                logs_filename=define_l2e_logs_filename(
                    campaign_name=campaign_name,
                    station_name=station_name,
                    start_time=event_info["start_time"],
                    end_time=event_info["end_time"],
                    rolling=rolling,
                    accumulation_interval=accumulation_interval,
                ),
                folder_partitioning=folder_partitioning,
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
            for event_info in files_partitions
        ]
        list_logs = execute_tasks_safely(list_tasks=list_tasks, parallel=parallel, logs_dir=logs_dir)

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
def define_l2m_logs_filename(campaign_name, station_name, start_time, end_time, model_name, sample_interval, rolling):
    """Define L2M logs filename."""
    temporal_resolution = define_temporal_resolution(seconds=sample_interval, rolling=rolling)
    starting_time = pd.to_datetime(start_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(end_time).strftime("%Y%m%d%H%M%S")
    logs_filename = (
        f"L2M_{model_name}.{temporal_resolution}.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}"
    )
    return logs_filename


@delayed_if_parallel
@single_threaded_if_parallel
def _generate_l2m(
    start_time,
    end_time,
    filepaths,
    data_dir,
    logs_dir,
    logs_filename,
    folder_partitioning,
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
    """Generate the L2M product from a DISDRODB L2E netCDF file."""
    # Define product
    product = "L2M"

    # Define product processing function
    def core(
        start_time,
        end_time,
        filepaths,
        campaign_name,
        station_name,
        # Processing options
        logger,
        verbose,
        force,
        # Product options
        product_options,
        sample_interval,
        rolling,
        model_name,
        # Archiving arguments
        data_dir,
        folder_partitioning,
    ):
        """Define L1 product processing."""
        # Copy to avoid in-place replacement (outside this function)
        product_options = product_options.copy()

        ##------------------------------------------------------------------------.
        # Extract L2M processing options
        l2m_options = product_options.get("product_options")
        radar_enabled = product_options.get("radar_enabled")
        radar_options = product_options.get("radar_options")

        # Define variables to load
        optimization_kwargs = l2m_options["optimization_kwargs"]
        if "init_method" in optimization_kwargs:
            init_method = optimization_kwargs["init_method"]
            moments = [f"M{order}" for order in init_method.replace("M", "")] + ["M1"]
        else:
            moments = ["M1"]

        variables = [
            "drop_number_concentration",
            "fall_velocity",
            "D50",
            "Nw",
            "Nt",
            "N",
            *moments,
        ]

        ##------------------------------------------------------------------------.
        # Open the netCDF files
        ds = open_netcdf_files(filepaths, start_time=start_time, end_time=end_time, variables=variables)
        ds = ds.load()
        ds.close()

        # Produce L2M dataset
        ds = generate_l2m(
            ds=ds,
            **l2m_options,
        )

        # Simulate L2M-based radar variables if asked
        if radar_enabled:
            ds_radar = generate_l2_radar(ds, parallel=not parallel, **radar_options)
            ds.update(ds_radar)
            ds.attrs = ds_radar.attrs.copy()  # ds_radar contains already all L2M attrs

        # Ensure at least 2 timestep available
        if ds["time"].size < 2:
            log_info(logger=logger, msg="File not created. Less than two timesteps available.", verbose=verbose)
            return None

        # Write L2M netCDF4 dataset
        filename = define_l2m_filename(
            ds,
            campaign_name=campaign_name,
            station_name=station_name,
            sample_interval=sample_interval,
            rolling=rolling,
            model_name=model_name,
        )
        folder_path = define_file_folder_path(ds, dir_path=data_dir, folder_partitioning=folder_partitioning)
        filepath = os.path.join(folder_path, filename)
        write_product(ds, filepath=filepath, force=force)

        # Return L2M dataset
        return ds

    # Define product processing function kwargs
    core_func_kwargs = dict(  # noqa: C408
        filepaths=filepaths,
        start_time=start_time,
        end_time=end_time,
        campaign_name=campaign_name,
        station_name=station_name,
        # Processing options
        verbose=verbose,
        force=force,
        # Product options
        product_options=product_options,
        sample_interval=sample_interval,
        rolling=rolling,
        model_name=model_name,
        # Archiving arguments
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

    # Check valid data_source, campaign_name, and station_name
    check_station_inputs(
        metadata_archive_dir=metadata_archive_dir,
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
    # temporal_resolution = "1MIN"
    # temporal_resolution = "10MIN"
    temporal_resolutions = get_product_temporal_resolutions("L2M")
    for temporal_resolution in temporal_resolutions:

        # Retrieve accumulation_interval and rolling option
        accumulation_interval, rolling = get_resampling_information(temporal_resolution)

        # ------------------------------------------------------------------.
        # Avoid generation of rolling products for source sample interval !
        if rolling and accumulation_interval == sample_interval:
            continue

        # Avoid product generation if the accumulation_interval is less than the sample interval
        if accumulation_interval < sample_interval:
            continue

        # -----------------------------------------------------------------.
        # List files to process
        # - If no data available, print error message and try with other L2E accumulation intervals
        required_product = get_required_product(product)
        filepaths = try_get_required_filepaths(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=required_product,
            # Processing options
            debugging_mode=debugging_mode,
            # Product options
            sample_interval=accumulation_interval,
            rolling=rolling,
        )
        if filepaths is None:
            continue

        # -------------------------------------------------------------------------.
        # Retrieve L2M processing options
        l2m_processing_options = ProcessingOptions(
            product="L2M",
            temporal_resolutions=temporal_resolution,
            filepaths=filepaths,
            parallel=parallel,
        )

        # Retrieve folder partitioning (for files and logs)
        folder_partitioning = l2m_processing_options.get_folder_partitioning(temporal_resolution)

        # Retrieve product options
        global_product_options = l2m_processing_options.get_product_options(temporal_resolution)

        # Retrieve files temporal partitions
        files_partitions = l2m_processing_options.get_files_partitions(temporal_resolution)

        if len(files_partitions) == 0:
            msg = (
                f"{product} processing of {data_source} {campaign_name} {station_name} "
                + f"has not been launched because of missing {required_product} {temporal_resolution} data."
            )
            log_info(logger=logger, msg=msg, verbose=verbose)
            continue

        # -----------------------------------------------------------------.
        # Loop over distributions to fit
        # model_name = "GAMMA_ML"
        # model_options =  l2m_options["models"][model_name]
        # Retrieve list of models to fit
        models = global_product_options.pop("models")
        for model_name in models:
            # -----------------------------------------------------------------.
            # Retrieve product-model options
            product_options = copy.deepcopy(global_product_options)
            model_options = get_model_options(product="L2M", model_name=model_name)
            product_options["product_options"].update(model_options)

            psd_model = model_options["psd_model"]
            optimization = model_options["optimization"]

            # Precompute required scattering tables
            if product_options["radar_enabled"]:
                radar_options = product_options["radar_options"]
                precompute_scattering_tables(verbose=verbose, **radar_options)

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
                    start_time=event_info["start_time"],
                    end_time=event_info["end_time"],
                    filepaths=event_info["filepaths"],
                    data_dir=data_dir,
                    logs_dir=logs_dir,
                    logs_filename=define_l2m_logs_filename(
                        campaign_name=campaign_name,
                        station_name=station_name,
                        start_time=event_info["start_time"],
                        end_time=event_info["end_time"],
                        model_name=model_name,
                        sample_interval=accumulation_interval,
                        rolling=rolling,
                    ),
                    folder_partitioning=folder_partitioning,
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
                for event_info in files_partitions
            ]
            list_logs = execute_tasks_safely(list_tasks=list_tasks, parallel=parallel, logs_dir=logs_dir)

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
