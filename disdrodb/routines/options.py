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
"""Implements ProcessingOption class for DISDRODB routines."""
import json
import os

import disdrodb
from disdrodb.api.checks import check_product, check_sensor_name, check_temporal_resolution
from disdrodb.api.info import group_filepaths
from disdrodb.configs import get_products_configs_dir
from disdrodb.utils.archiving import define_temporal_partitions, group_files_by_temporal_partitions
from disdrodb.utils.list import flatten_list
from disdrodb.utils.routines import is_possible_product
from disdrodb.utils.time import ensure_timedelta_seconds, get_sampling_information
from disdrodb.utils.yaml import read_yaml

# TODO: Test ensure recursive update for product_options key, do not replace just "product_options" dict !
# get_product_options(product="L2E", temporal_resolution="10MIN")
# get_product_options(product="L2M", temporal_resolution="10MIN")
# get_product_options(product="L1")
# get_product_options(product="L1", temporal_resolution="1MIN")
# get_product_options(product="L1", temporal_resolution="1MIN", sensor_name="PARSIVEL")

# test temporal_resolutions are unique

# TODO: test return list
# get_product_temporal_resolutions(product="L1")
# get_product_temporal_resolutions(product="L2E")
# get_product_temporal_resolutions(product="L2M")


def get_product_options(product, temporal_resolution=None, sensor_name=None):
    """Return DISDRODB product options.

    If temporal resolution is not provided, it returns the global product option.
    If temporal resolution is provided, it provides the custom product options, if specified.
    If product="L1" and sensor_name is specified, it customize product options also by sensor.
    """
    # Retrieve products configuration directory
    products_configs_dir = get_products_configs_dir()

    # Validate DISDRODB products configuration
    validate_product_configuration(products_configs_dir)

    # Check product
    check_product(product)

    # Retrieve global product options (when no temporal resolution !)
    global_options = read_yaml(os.path.join(products_configs_dir, product, "global.yaml"))
    if temporal_resolution is None:
        global_options = check_availability_radar_simulations(global_options)
        return global_options

    # Check temporal resolution
    check_temporal_resolution(temporal_resolution)

    # If temporal resolutions are specified, drop 'temporal_resolutions' key
    global_options.pop("temporal_resolutions", None)

    # Read custom options for specific temporal resolution
    custom_options_path = os.path.join(products_configs_dir, product, f"{temporal_resolution}.yaml")
    if not os.path.exists(custom_options_path):
        return global_options
    custom_options = read_yaml(custom_options_path)

    # Define product options
    options = global_options.copy()
    if "product_options" in custom_options:
        options["product_options"].update(custom_options.pop("product_options"))
    options.update(custom_options)

    # Check availability of radar simulations
    options = check_availability_radar_simulations(options)

    # Customize product options by sensor if L1 product
    if product == "L1" and sensor_name is not None:
        check_sensor_name(sensor_name)
        custom_options_path = os.path.join(products_configs_dir, product, sensor_name, f"{temporal_resolution}.yaml")
        if not os.path.exists(custom_options_path):
            return options
        custom_options = read_yaml(custom_options_path)
        if "product_options" in custom_options:
            options["product_options"].update(custom_options.pop("product_options"))
    return options


def get_product_temporal_resolutions(product):
    """Return DISDRODB products temporal resolutions."""
    # Check only L2E and L2M
    return get_product_options(product)["temporal_resolutions"]


def get_model_options(product, model_name):
    """Return DISDRODB L2M product model options."""
    # Retrieve products configuration directory
    products_configs_dir = get_products_configs_dir()
    model_options_path = os.path.join(products_configs_dir, product, "MODELS", f"{model_name}.yaml")
    model_options = read_yaml(model_options_path)
    return model_options


def check_availability_radar_simulations(options):
    """Check radar simulations are possible for L2E and L2M products."""
    if "radar_enabled" in options and not disdrodb.is_pytmatrix_available():
        options["radar_enabled"] = False
    return options


def validate_product_configuration(products_configs_dir):
    """Validate the DISDRODB products configuration files."""
    # TODO: Implement validation of DISDRODB products configuration files with pydantic
    # TODO: Raise warning if L1 temporal resolutions does not includes all temporal resolutions of L2 products.
    # TODO: Raise warning if L2E temporal resolutions does not includes all temporal resolutions of L2M products.
    # if stategy_event, check neighbor_time_interval >= sample_interval !
    # if temporal_resolution_to_seconds(neighbor_time_interval) < temporal_resolution_to_seconds(sample_interval):
    #     msg = "'neighbor_time_interval' must be at least equal to the dataset sample interval ({sample_interval})"
    #     raise ValueError(msg)

    pass


def _define_blocks_offsets(sample_interval, temporal_resolution):
    """Define blocks offset for resampling logic."""
    # Retrieve accumulation_interval and rolling option
    accumulation_interval, rolling = get_sampling_information(temporal_resolution)

    # Ensure sample_interval and accumulation_interval is numpy.timedelta64
    accumulation_interval = ensure_timedelta_seconds(accumulation_interval)
    sample_interval = ensure_timedelta_seconds(sample_interval)

    # Define offset to apply to time partitions blocks
    block_starts_offset = 0
    block_ends_offset = 0

    # If rolling, need to search also in next time block ...
    if rolling and sample_interval != accumulation_interval:
        block_ends_offset = accumulation_interval - sample_interval
    return block_starts_offset, block_ends_offset


class L1ProcessingOptions:
    """Define L1 product processing options."""

    def __init__(self, filepaths, parallel, sensor_name, temporal_resolutions=None):
        """Define DISDRODB L1 product processing options."""
        product = "L1"

        # ---------------------------------------------------------------------.
        # Define temporal resolutions for which to retrieve processing options
        if temporal_resolutions is None:
            temporal_resolutions = get_product_temporal_resolutions(product)
        elif isinstance(temporal_resolutions, str):
            temporal_resolutions = [temporal_resolutions]
        _ = [check_temporal_resolution(temporal_resolution) for temporal_resolution in temporal_resolutions]

        # ---------------------------------------------------------------------.
        # Get product options at various temporal resolutions
        src_dict_product_options = {
            temporal_resolution: get_product_options(
                product=product,
                temporal_resolution=temporal_resolution,
                sensor_name=sensor_name,
            )
            for temporal_resolution in temporal_resolutions
        }

        # ---------------------------------------------------------------------.
        # Group filepaths by source sample intervals
        # - Typically the sample interval is fixed and is just one
        # - Some stations might change the sample interval along the years
        # - For each sample interval, separated processing must take place
        dict_filepaths = group_filepaths(filepaths, groups="sample_interval")

        # ---------------------------------------------------------------------.
        # Retrieve processing information for each temporal resolution
        dict_product_options = {}
        dict_folder_partitioning = {}
        dict_files_partitions = {}
        _cache_dict_temporal_partitions: dict[str, dict] = {}
        # temporal_resolution = temporal_resolutions[0]
        for temporal_resolution in temporal_resolutions:

            # -------------------------------------------------------------------------.
            # Retrieve product options
            product_options = src_dict_product_options[temporal_resolution].copy()

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
            # --> Here we cache dict_temporal_partitions so that we don't need to recompute
            #     stuffs if processing options are the same
            # --> Using time_block with e.g. freq="day" we get start/end in the format of 00:00:00-23:59:59
            key = json.dumps(archive_options, sort_keys=True)
            if key not in _cache_dict_temporal_partitions:
                _cache_dict_temporal_partitions[key] = {
                    sample_interval: define_temporal_partitions(filepaths, parallel=parallel, **archive_options)
                    for sample_interval, filepaths in dict_filepaths.items()
                }
            dict_temporal_partitions = _cache_dict_temporal_partitions[key].copy()  # To avoid in-place replacement

            # ------------------------------------------------------------------.
            # Group filepaths by temporal partitions
            # - This is done separately for each possible source sample interval
            # - It groups filepaths by start_time and end_time provided by temporal_partitions
            #   --> Output: [{'start_time': ...,  'end_time': ..., filepaths: [...]}, ...]
            # - In L0C we ensure that the time reported correspond to the start of the measurement interval.
            # - When aggregating/resampling/accumulating data, we need to load also some data after the
            #   actual end_time of the time partition to ensure that
            #   the resampled dataset contains the timesteps of the partition end time.
            #   --> Use of block_starts_offset and block_ends_offset in group_files_by_temporal_partitions
            # - ATTENTION: group_files_by_temporal_partitions returns
            #              start_time and end_time as datetime.datetime64objects !
            # - ATTENTION: Files within each files_partitions block have the same sample_interval !

            # sample_interval = 30
            # temporal_partitions = dict_temporal_partitions[sample_interval]

            files_partitions = []
            for sample_interval, temporal_partitions in dict_temporal_partitions.items():
                if is_possible_product(
                    temporal_resolution=temporal_resolution,
                    sample_interval=sample_interval,
                ):

                    block_starts_offset, block_ends_offset = _define_blocks_offsets(
                        sample_interval=sample_interval,
                        temporal_resolution=temporal_resolution,
                    )

                    files_partitions.append(
                        group_files_by_temporal_partitions(
                            temporal_partitions=temporal_partitions,
                            filepaths=dict_filepaths[sample_interval],
                            block_starts_offset=block_starts_offset,
                            block_ends_offset=block_ends_offset,
                        ),
                    )
            files_partitions = flatten_list(files_partitions)
            dict_files_partitions[temporal_resolution] = files_partitions

        # ------------------------------------------------------------------.
        # Keep only temporal_resolutions for which products can be defined
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

    def group_files_by_temporal_partitions(self, temporal_resolution):
        """Return files partitions dictionary for a specific L2E product."""
        return self.dict_files_partitions[temporal_resolution]

    def get_product_options(self, temporal_resolution):
        """Return product options dictionary for a specific L2E product."""
        return self.dict_product_options[temporal_resolution]

    def get_folder_partitioning(self, temporal_resolution):
        """Return the folder partitioning for a specific L2E product."""
        # to be used for logs and files !
        return self.dict_folder_partitioning[temporal_resolution]


class L2ProcessingOptions:
    """Define L2 products processing options."""

    def __init__(self, product, filepaths, parallel, temporal_resolution):
        """Define DISDRODB L2 products processing options."""
        import disdrodb

        # Check temporal resolution
        check_temporal_resolution(temporal_resolution)

        # Get product options
        product_options = get_product_options(product, temporal_resolution=temporal_resolution)

        # Extract processing options
        archive_options = product_options.pop("archive_options")

        # Define folder partitioning
        if "folder_partitioning" not in archive_options:
            folder_partitioning = disdrodb.config.get("folder_partitioning")
        else:
            folder_partitioning = archive_options.pop("folder_partitioning")

        # Define files temporal partitions
        # - [{start_time: np.datetime64, end_time: np.datetime64}, ....]
        # - Either strategy: "event" or "time_block"
        # - "strategy=event" requires loading data into memory to identify events
        #   --> Does some data filtering on what to process !
        # - "strategy=time_block" does not require loading data into memory
        #   --> Does not do data filtering on what to process !
        temporal_partitions = define_temporal_partitions(filepaths, parallel=parallel, **archive_options)

        # ------------------------------------------------------------------.
        # Group filepaths by temporal partitions
        # - It groups filepaths by start_time and end_time provided by temporal_partitions
        # - ATTENTION: group_files_by_temporal_partitions returns
        #              start_time and end_time as datetime.datetime64 objects !
        files_partitions = group_files_by_temporal_partitions(
            temporal_partitions=temporal_partitions,
            filepaths=filepaths,
        )
        files_partitions = flatten_list(files_partitions)

        # ------------------------------------------------------------------.
        # Add attributes
        # self.temporal_partitions = temporal_partitions
        self.folder_partitioning = folder_partitioning
        self.files_partitions = files_partitions
        self.product_options = product_options
