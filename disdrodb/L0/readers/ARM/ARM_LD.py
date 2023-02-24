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
"""Reader for ARM OTT Parsivel sensor."""
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
        "particle_size": "diameter_bin_center",
        "raw_fall_velocity": "velocity_bin_center",
        ### Variables
        # 'base_time': 'base_time',
        # 'time_offset'
        # 'class_size_width',
        # 'fall_velocity_calculated',
        # "lat": "latitude",
        # "lon": "longitude",
        # "alt": "altitude",
        "weather_code": "weather_code_synop_4680",
        "equivalent_radar_reflectivity_ott": "reflectivity_32bit",
        "mor_visibility": "mor_visibility",
        "sensor_temperature": "sensor_temperature",
        "laserband_amplitude": "laser_amplitude",
        "heating_current": "sensor_heating_current",
        "sensor_voltage": "sensor_battery_voltage",
        "number_detected_particles": "number_particles",
        "snow_depth_intensity": "snowfall_rate",  # Only available > 2019
        "number_density_drops": "raw_drop_concentration",
        "raw_spectrum": "raw_drop_number",
        ### ARM retrievals
        # 'moment1',
        # 'moment2',
        # 'moment3',
        # 'moment4',
        # 'moment5',
        # 'moment6',
        # 'diameter_min':
        # 'diameter_max':
        # 'median_volume_diameter'
        # 'intercept_parameter',
        # 'slope_parameter'
        # 'liquid_water_content',
        # 'liquid_water_distribution_mean'
        # 'precip_rate':
        # 'equivalent_radar_reflectivity',
        # Possible QC variables
        # 'qc_time',
        # 'qc_precip_rate':
        # 'qc_number_detected_particles':
        # 'qc_mor_visibility':
        # 'qc_heating_current':
        # 'qc_snow_depth_intensity':
        # 'qc_laserband_amplitude':
        # 'qc_weather_code':
        # 'qc_equivalent_radar_reflectivity_ott':
        # 'qc_sensor_voltage':
    }

    # Define dataset sanitizer
    def ds_sanitizer_fun(ds):
        from disdrodb.l0.l0b_processing import replace_custom_nan_flags

        # Replace nan flags with np.nan
        # - ARM use the -9999 flags
        nan_flags_variables = [
            "sensor_temperature",
            "laser_amplitude",
            "mor_visibility",
            "number_particles",
            "weather_code_synop_4680",
            "raw_drop_number",
        ]
        dict_nan_flags = {var: [-9999] for var in nan_flags_variables}
        ds = replace_custom_nan_flags(ds, dict_nan_flags=dict_nan_flags)

        # Return dataset
        return ds

    # Define glob pattern to search data files in <raw_dir>/data/<station_name>
    glob_patterns = "*.cdf"

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
