#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from disdrodb.l0 import run_l0b_from_nc
from disdrodb.l0.l0_reader import reader_generic_docstring, is_documented_by


@is_documented_by(reader_generic_docstring)
def reader(
    raw_dir,
    processed_dir,
    station_name,
    # Processing options
    force=False,
    verbose=False,
    parallel=False,
    debugging_mode=False,
):
    # Define dictionary mapping dataset variables to select and rename
    dict_names = {
        ### Dimensions
        "diameter": "diameter_bin_center",
        "fallspeed": "velocity_bin_center",
        ### Variables
        "precipitation_flux": "precipitation_rate",
        "solid_precipitation_flux": "snowfall_rate",
        "precipitation_visibility": "mor_visibility",
        "reflectivity": "reflectivity",
        "present_weather_1m": "weather_code_synop_4680",
        "present_weather_5m": "weather_code_synop_4680_5min",
        "max_hail_diameter": "max_hail_diameter",
        "particle_count": "number_particles",
        # "measurement_quality": "quality_index",
        ### Arrays
        # "drop_size_distribution": "raw_drop_concentration",
        # "drop_velocity_distribution": "raw_drop_average_velocity",
        "size_velocity_distribution": "raw_drop_number",
        ### Variables to discard
        # 'year'
        # 'month'
        # 'day'
        # 'hour'
        # 'minute'
        # 'second'
        # 'day_of_year'
        # 'hydrometeor_type_1m' # Pickering et al., 2019
        # 'hydrometeor_type_5m' # Pickering et al., 2019
    }

    # Define dataset sanitizer
    def ds_sanitizer_fun(ds):
        # Drop coordinates not DISDRODB-compliants
        pass
        # Return dataset
        return ds

    # Define glob pattern to search data files in <raw_dir>/data/<station_name>
    glob_patterns = "*.nc*"

    ####----------------------------------------------------------------------.
    #### - Create L0A products
    run_l0b_from_nc(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        station_name=station_name,
        # Custom arguments of the reader
        glob_patterns=glob_patterns,
        dict_names=dict_names,
        ds_sanitizer_fun=ds_sanitizer_fun,
        # Processing options
        force=force,
        verbose=verbose,
        parallel=parallel,
        debugging_mode=debugging_mode,
    )


# -----------------------------------------------------------------.
