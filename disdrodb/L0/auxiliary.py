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


def get_DELFT_dict(sensor_name: str) -> dict:
    """
    Dictionary mapping from DELFT standards to DISDRODB standards.

    Parameters
    ----------
    sensor_name : str
        Disdrometer sensor name.

    """
    if sensor_name == "OTT_Parsivel":
        d = {
            "Meas_Time": "time",  # TO BE CHECKED
            "Meas_Interval": "sample_interval",
            "RR_Intensity": "rainfall_rate_32bit",
            "RR_Accumulated": "rainfall_accumulated_32bit",
            "RR_Total": "rainfall_amount_absolute_32bit",
            "Synop_WaWa": "weather_code_synop_4680",
            "Synop_WW": "weather_code_synop_4677",
            "Reflectivity": "reflectivity_32bit",
            "Visibility": "mor_visibility",
            "T_Sensor": "sensor_temperature",
            "Sig_Laser": "laser_amplitude",
            "N_Particles": "number_particles",
            "State_Sensor": "sensor_status",
            "E_kin": "rain_kinetic_energy",
            "V_Sensor": "sensor_battery_voltage",
            "I_Heating": "sensor_heating_current",
            "Error_Code": "error_code",
            "Data_Raw": "raw_drop_number",
            "Data_N_Field": "raw_drop_concentration",
            "Data_V_Field": "raw_drop_average_velocity",
        }
    else:
        raise NotImplementedError("DELFT standard implemented only for OTT Parsivel")
    return d


def get_DIVEN_dict(sensor_name: str) -> dict:
    """
    Dictionary mapping from DIVEN standards to DISDRODB standards.

    Parameters
    ----------
    sensor_name : str
        Disdrometer sensor name.

    """
    d = {
        ## Time is to decode
        # 'year'
        # 'month'
        # 'day'
        # 'hour'
        # 'minute'
        # 'second'
        # 'day_of_year'
        "precipitation_flux": "precipitation_rate",
        "solid_precipitation_flux": "snowfall_rate",
        "precipitation_visibility": "mor_visibility",
        "reflectivity": "reflectivity",
        "present_weather_1m": "weather_code_synop_4680",
        "present_weather_5m": "weather_code_synop_4680_5min",
        "max_hail_diameter": "max_hail_diameter",
        "particle_count": "number_particles",
        # 'hydrometeor_type_1m' # Pickering et al., 2019
        # 'hydrometeor_type_5m' # Pickering et al., 2019
        # "measurement_quality": "quality_index",
        # Arrays
        "size_velocity_distribution": "raw_drop_number",
        # "drop_size_distribution": "raw_drop_concentration",
        # "drop_velocity_distribution": "raw_drop_average_velocity",
    }
    return d


def get_ARM_LPM_dict(sensor_name: str) -> dict:
    """
    Dictionary mapping from ARM standards to DISDRODB standards.

    Parameters
    ----------
    sensor_name : str
        Disdrometer sensor name.

    """
    if sensor_name == "Thies_LPM":
        # Dimensions:
        # 'time': 'time # to use
        # "particle_diameter": "diameter bin id ",
        # "particle_fall_velocity": "velocity_bin_id",
        d = {
            # 'base_time': 'base_time',
            # 'time_offset': 'time_offset',
            # 'time_bounds': 'time_bounds',
            # "particle_diameter_bounds",
            # "particle_fall_velocity_bounds"
            "lat": "latitude",
            "lon": "longitude",
            "alt": "altitude",
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
    elif sensor_name == "OTT_Parsivel2":
        # Coords
        # - time
        # - raw_fall_velocity
        # - particle_size
        d = {
            # 'base_time': 'base_time',
            # 'time_offset'
            # 'class_size_width',
            # 'fall_velocity_calculated',
            "lat": "latitude",
            "lon": "longitude",
            "alt": "altitude",
            "weather_code": "weather_code_synop_4680",
            "equivalent_radar_reflectivity_ott": "reflectivity_32bit",
            "mor_visibility": "mor_visibility",
            "sensor_temperature": "sensor_temperature",
            "laserband_amplitude": "laser_amplitude",
            "heating_current": "sensor_heating_current",
            "sensor_voltage": "sensor_battery_voltage",
            "number_detected_particles": "number_particles",
            "raw_spectrum": "raw_drop_number",
            "number_density_drops": "raw_drop_concentration",
            "snow_depth_intensity": "snowfall_rate",  # Only available > 2019
            # ARM retrievals
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
    else:
        raise NotImplementedError
    return d
