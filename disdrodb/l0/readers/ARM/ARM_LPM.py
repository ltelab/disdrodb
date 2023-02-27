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
"""Reader for ARM Thies LPM sensor."""
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
    dict_names = {
        ## Dimensions
        "particle_diameter": "diameter_bin_center",
        "particle_fall_velocity": "velocity_bin_center",
        ### Variables
        # "lat": "latitude",
        # "lon": "longitude",
        # "alt": "altitude",
        # 'base_time': 'base_time',
        # 'time_offset': 'time_offset',
        # 'time_bounds': 'time_bounds',
        # "particle_diameter_bounds",
        # "particle_fall_velocity_bounds"
        "synop_4677_weather_code": "weather_code_synop_4677",
        "metar_4678_weather_code": "weather_code_metar_4678",
        "synop_4680_weather_code": "weather_code_synop_4680",
        "synop_4677_5min_weather_code": "weather_code_synop_4677_5min",
        "metar_4678_5min_weather_code": "weather_code_metar_4678_5min",
        "synop_4680_5min_weather_code": "weather_code_synop_4680_5min",
        "intensity_total_5min": "precipitation_rate_5min",
        "intensity_total": "precipitation_rate",
        "intensity_liquid": "rainfall_rate",
        "intensity_solid": "snowfall_rate",
        "accum_precip": "precipitation_accumulated",
        "maximum_visibility": "mor_visibility",
        "radar_reflectivity": "reflectivity",
        "quality_measurement": "quality_index",
        "max_diameter_hail": "max_hail_diameter",
        "laser_status": "laser_status",
        "static_signal": "static_signal",
        "interior_temperature": "temperature_interior",
        "laser_temperature": "laser_temperature",
        "laser_temperature_analog_status": "laser_temperature_analog_status",
        "laser_temperature_digital_status": "laser_temperature_digital_status",
        "mean_laser_current": "laser_current_average",
        "laser_current_analog_status": "laser_current_analog_status",
        "laser_current_digital_status": "laser_current_digital_status",
        "control_voltage": "control_voltage",
        "optical_control_output": "optical_control_voltage_output",
        "control_output_laser_power_status": "control_output_laser_power_status",
        "voltage_sensor_supply": "sensor_voltage_supply",
        "voltage_sensor_supply_status": "sensor_voltage_supply_status",
        "ambient_temperature": "temperature_ambient",
        "temperature_sensor_status": "temperature_sensor_status",
        "voltage_heating_supply": "current_heating_voltage_supply",
        "voltage_heating_supply_status": "current_heating_voltage_supply_status",
        "pane_heating_laser_head_current": "current_heating_pane_transmitter_head",
        "pane_heating_laser_head_current_status": "current_heating_pane_transmitter_head_status",
        "pane_heating_receiver_head_current": "current_heating_pane_receiver_head",
        "pane_heating_receiver_head_current_status": "current_heating_pane_receiver_head_status",
        "heating_house_current": "current_heating_house",
        "heating_house_current_status": "current_heating_house_status",
        "heating_heads_current": "current_heating_heads",
        "heating_heads_current_status": "current_heating_heads_status",
        "heating_carriers_current": "current_heating_carriers",
        "heating_carriers_current_status": "current_heating_carriers_status",
        "number_particles": "number_particles",
        "number_particles_internal_data": "number_particles_internal_data",
        "number_particles_min_speed": "number_particles_min_speed",
        "number_particles_min_speed_internal_data": "number_particles_min_speed_internal_data",
        "number_particles_max_speed": "number_particles_max_speed",
        "number_particles_max_speed_internal_data": "number_particles_max_speed_internal_data",
        "number_particles_min_diameter": "number_particles_min_diameter",
        "number_particles_min_diameter_internal_data": "number_particles_min_diameter_internal_data",
        "precipitation_spectrum": "raw_drop_number",
        # 'air_temperature',
    }

    # Define dataset sanitizer
    def ds_sanitizer_fun(ds):
        from disdrodb.l0.l0b_processing import replace_custom_nan_flags

        # Replace nan flags with np.nan
        # - ARM use the -9999 flags
        nan_flags_variables = [
            "weather_code_synop_4677_5min",
            "weather_code_synop_4680_5min",
            "weather_code_synop_4680",
            "weather_code_synop_4677",
            "mor_visibility",
            "quality_index",
            "temperature_interior",
            "laser_temperature",
            "laser_current_average",
            "number_particles",
            "number_particles_min_speed",
            "number_particles_max_speed",
            "number_particles_min_diameter",
            "number_particles_max_diameter",
            "raw_drop_number",
        ]
        dict_nan_flags = {var: [-9999] for var in nan_flags_variables}
        ds = replace_custom_nan_flags(ds, dict_nan_flags=dict_nan_flags)

        # Return dataset
        return ds

    # Define glob pattern to search data files in <raw_dir>/data/<station_name>
    glob_patterns = "*.nc"

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
