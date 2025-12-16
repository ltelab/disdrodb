# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
from pathlib import Path

import disdrodb
from disdrodb.api.checks import check_product, check_temporal_resolution
from disdrodb.api.info import group_filepaths
from disdrodb.configs import get_products_configs_dir
from disdrodb.utils.archiving import define_temporal_partitions, group_files_by_temporal_partitions
from disdrodb.utils.list import flatten_list
from disdrodb.utils.routines import is_possible_product
from disdrodb.utils.time import ensure_timedelta_seconds, get_sampling_information
from disdrodb.utils.yaml import read_yaml


def get_product_options_directory(products_configs_dir, product, sensor_name=None):
    """Retrieve path to the product options directory."""
    products_configs_dir = str(products_configs_dir)  # convert pathlib to str
    if sensor_name is None:
        return os.path.join(products_configs_dir, product)
    return os.path.join(products_configs_dir, product, sensor_name)


def get_l2m_model_settings_directory(products_configs_dir):
    """Retrieve path to the product options directory."""
    return os.path.join(products_configs_dir, "L2M", "MODELS")


def get_l2m_model_settings_files(products_configs_dir):
    """Retrieve path to the product options directory."""
    models_dir = Path(get_l2m_model_settings_directory(products_configs_dir))
    models_files = list(models_dir.glob("*.yaml")) + list(models_dir.glob("*.yml"))
    return models_files


def get_product_global_options_path(products_configs_dir, product, sensor_name=None):
    """Retrieve path to the product global options."""
    product_options_dir = get_product_options_directory(
        products_configs_dir=products_configs_dir,
        product=product,
        sensor_name=sensor_name,
    )
    product_global_options_path = os.path.join(product_options_dir, "global.yaml")
    if os.path.exists(product_global_options_path):
        return product_global_options_path

    product_options_dir = get_product_options_directory(
        products_configs_dir=products_configs_dir,
        product=product,
        sensor_name=None,
    )
    global_options_path = os.path.join(product_options_dir, "global.yaml")  # this must exists
    return global_options_path


def get_product_custom_options_path(products_configs_dir, product, temporal_resolution, sensor_name=None):
    """Retrieve path to the product temporal resolution custom options."""
    product_options_dir = get_product_options_directory(
        products_configs_dir=products_configs_dir,
        product=product,
        sensor_name=sensor_name,
    )
    product_global_options_path = os.path.join(product_options_dir, f"{temporal_resolution}.yaml")
    if os.path.exists(product_global_options_path):
        return product_global_options_path

    product_options_dir = get_product_options_directory(
        products_configs_dir=products_configs_dir,
        product=product,
        sensor_name=None,
    )
    custom_options_path = os.path.join(product_options_dir, f"{temporal_resolution}.yaml")  # this might not exists
    return custom_options_path


def get_product_options(product, temporal_resolution=None, sensor_name=None, products_configs_dir=None):
    """Return DISDRODB product options.

    If temporal resolution is not provided, it returns the global product option.
    If temporal resolution is provided, it provides the custom product options, if specified.
    If product="L1" and sensor_name is specified, it customize product options also by sensor.
    """
    # Retrieve products configuration directory
    products_configs_dir = get_products_configs_dir(products_configs_dir=products_configs_dir)

    # Check product
    check_product(product)

    # Get product global options path
    global_options_path = get_product_global_options_path(
        products_configs_dir=products_configs_dir,
        product=product,
        sensor_name=sensor_name,
    )

    # Retrieve global product options (when no temporal resolution !)
    global_options = read_yaml(global_options_path)
    if temporal_resolution is None:
        global_options = check_availability_radar_simulations(global_options)
        return global_options

    # Check temporal resolution
    check_temporal_resolution(temporal_resolution)

    # If temporal resolutions are specified, drop 'temporal_resolutions' key
    global_options.pop("temporal_resolutions", None)

    # Read custom options for specific temporal resolution
    custom_options_path = get_product_custom_options_path(
        products_configs_dir=products_configs_dir,
        product=product,
        sensor_name=sensor_name,
        temporal_resolution=temporal_resolution,
    )
    if not os.path.exists(custom_options_path):
        return global_options
    custom_options = read_yaml(custom_options_path)

    # Update global options with the custom options specified
    options = global_options.copy()
    keys_options = ["archive_options", "product_options", "radar_options"]
    for key in keys_options:
        if key in custom_options:
            options[key].update(custom_options.pop(key))  # Update with only one

    # Update remaining flat keys
    options.update(custom_options)

    # Check availability of radar simulations
    options = check_availability_radar_simulations(options)

    return options


def get_product_temporal_resolutions(product, sensor_name=None):
    """Return DISDRODB products temporal resolutions."""
    # Check only L2E and L2M
    return get_product_options(product, sensor_name=sensor_name)["temporal_resolutions"]


def get_model_options(model_name, products_configs_dir=None):
    """Return DISDRODB L2M product model options."""
    # Retrieve products configuration directory
    products_configs_dir = get_products_configs_dir(products_configs_dir=products_configs_dir)
    models_settings_dir = get_l2m_model_settings_directory(products_configs_dir)
    model_options_path = os.path.join(models_settings_dir, f"{model_name}.yaml")
    model_options = read_yaml(model_options_path)
    return model_options


def check_availability_radar_simulations(options):
    """Check radar simulations are possible for L2E and L2M products."""
    if "radar_enabled" in options and not disdrodb.is_pytmatrix_available():
        options["radar_enabled"] = False
    return options


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


class L0CProcessingOptions:
    """Define L0C product processing options."""

    def __init__(self, sensor_name):
        """Define DISDRODB L0C product processing options."""
        product = "L0C"
        options = get_product_options(product=product, sensor_name=sensor_name)["archive_options"]

        self.product = product
        self.folder_partitioning = options["folder_partitioning"]
        self.product_frequency = options["strategy_options"]["freq"]


class L1ProcessingOptions:
    """Define L1 product processing options."""

    def __init__(self, filepaths, parallel, sensor_name, temporal_resolutions=None):
        """Define DISDRODB L1 product processing options."""
        product = "L1"

        # ---------------------------------------------------------------------.
        # Define temporal resolutions for which to retrieve processing options
        if temporal_resolutions is None:
            temporal_resolutions = get_product_temporal_resolutions(product, sensor_name=sensor_name)
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

            # Define folder partitioning
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
        """Return files partitions dictionary for a specific L1 product."""
        return self.dict_files_partitions[temporal_resolution]

    def get_product_options(self, temporal_resolution):  # noqa
        """Return product options dictionary for a specific L1 product."""
        return {}  # self.dict_product_options[temporal_resolution]

    def get_folder_partitioning(self, temporal_resolution):
        """Return the folder partitioning for a specific L1 product."""
        return self.dict_folder_partitioning[temporal_resolution]


class L2ProcessingOptions:
    """Define L2 products processing options."""

    def __init__(self, product, filepaths, parallel, temporal_resolution, sensor_name):
        """Define DISDRODB L2 products processing options."""
        # Check temporal resolution
        check_temporal_resolution(temporal_resolution)

        # Get product options
        product_options = get_product_options(product, temporal_resolution=temporal_resolution, sensor_name=sensor_name)

        # Extract processing options
        archive_options = product_options.pop("archive_options")

        # Define folder partitioning
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
