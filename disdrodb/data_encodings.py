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
def get_LPM_variables():
    # '01': start_identifier
    # '02': device_address
    # '03': sensor_serial_number
    # '04': software_version
    # '05': date_sensor
    # '06': time_sensor

    # '07': weather_code_synop_4677_5min
    # '08': weather_code_synop_4680_5min
    # '09': weather_code_metar_4678_5min
    # '10': precipitation_rate_5min
    # '11': weather_code_synop_4677
    # '12': weather_code_synop_4680
    # '13': weather_code_metar_4678
    # '14': precipitation_rate  # intensity_total
    # '15': rainfall_rate # intensity_liquid
    # '16': snowfall_rate # intensity_solid
    # '17': precipitation_accumulated  # accum_precip
    # '18': mor_visibility  # maximum_visibility
    # '19': reflectivity # radar_reflectivity
    # '20': quality_index # [0-100]
    # '21': max_hail_diameter

    # # status
    # '22': laser_status
    # '23': static_signal

    # '24': laser_temperature_analog_status
    # '25': laser_temperature_digital_status
    # '26': laser_current_analog_status
    # '27': laser_current_digital_status

    # # Heating: glass(panes), housing, heads, carriers
    # '28': sensor_voltage_supply_status

    # '29': current_heating_pane_transmitter_head_status
    # '30': current_heating_pane_receiver_head_status

    # '31': temperature_sensor_status

    # '32': current_heating_voltage_supply_status
    # '33': current_heating_house_status
    # '34': current_heating_heads_status
    # '35': current_heating_carriers_status

    # '36': control_output_laser_power_status
    # '37': reserve_status

    # # Values
    # '38': temperature_interior

    # '39': laser_temperature
    # '40': laser_current_average

    # '41': control_voltage
    # '42': optical_control_voltage_output # maybe think a bit more on a nicer name # control_output_laser_voltage

    # '43': sensor_voltage_supply

    # '44': current_heating_pane_transmitter_head
    # '45': current_heating_pane_receiver_head

    # '46': temperature_ambient

    # '47': current_heating_voltage_supply # V
    # '48': current_heating_house # A
    # '49': current_heating_heads # A
    # '50': current_heating_carriers # A

    # '51': number_particles # n_particles_all
    # '52': number_particles_internal_data
    # '53': number_particles_min_speed
    # '54': number_particles_min_speed_internal_data
    # '55': number_particles_max_speed
    # '56': number_particles_max_speed_internal_data
    # '57': number_particles_min_diameter
    # '58': number_particles_min_diameter_internal_data
    # '59': number_particles_no_hydrometeor
    # '60': number_particles_no_hydrometeor_internal_data
    # '61': number_particles_unknown_classification
    # '62': number_particles_unknown_classification_internal_data
    # '63': number_particles_class_1
    # '64': number_particles_class_1_internal_data
    # '65': number_particles_class_2
    # '66': number_particles_class_2_internal_data
    # '67': number_particles_class_3
    # '68': number_particles_class_3_internal_data
    # '69': number_particles_class_4
    # '70': number_particles_class_4_internal_data
    # '71': number_particles_class_5
    # '72': number_particles_class_5_internal_data
    # '73': number_particles_class_6
    # '74': number_particles_class_6_internal_data
    # '75': number_particles_class_7
    # '76': number_particles_class_7_internal_data
    # '77': number_particles_class_8
    # '78': number_particles_class_8_internal_data
    # '79': number_particles_class_9
    # '80': number_particles_class_9_internal_data
    # '81': raw_spectrum
    # '520': diameter_speed
    return


def get_L0_dtype_standards(sensor_name: str) -> dict:
    from disdrodb.standards import get_L0_dtype

    # TODO: TO REFACTOR !!!!
    dtype_dict = {
        # # Disdronet raspberry variables
        # "epoch_time": "float32",
        # "time": "M8[s]",
        # "id": "uint32",
        # # Datalogger variables
        # "datalogger_heating_current": "float32",
        # "datalogger_battery_voltage": "float32",
        # "datalogger_temperature": "object",
        # "datalogger_voltage": "object",
        # "datalogger_error": "uint8",
        # # Coords
        # "latitude": "float32",
        # "longitude": "float32",
        # "altitude": "float32",
        # # Custom fields
        # "Unknow_column": "object",
        # # Temp variables
        # "temp": "object",
        # "temp1": "object",
        # "temp2": "object",
        # "temp3": "object",
        # "temp4": "object",
        
        # "TEMPORARY": "object",
        # "TO_BE_MERGE": "object",
        # "TO_BE_MERGE2": "object",
        # "TO_BE_PARSED": "object",
        # "TO_BE_SPLITTED": "object",
        # "TO_DEBUG": "object",
        # "Debug_data": "object",
        # "All_0": "object",
        # "error_code?": "object",
        # "unknow2": "object",
        # "unknow3": "object",
        # "unknow4": "object",
        # "unknow5": "object",
        # "unknow": "object",
        # "unknow6": "object",
        # "unknow7": "object",
        # "unknow8": "object",
        # "unknow9": "object",
        # "power_supply_voltage": "object",
        # "A_voltage2?": "object",
        # "A_voltage?": "object",
        # "All_nan": "object",
        # "All_5000": "object",
    }
    d1 = get_L0_dtype(sensor_name=sensor_name)
    dtype_dict.update(d1)
    return dtype_dict


def get_DIVEN_dict():
    d = {
        "precipitation_flux": "precipitation_rate",
        "solid_precipitation_flux": "snowfall_rate",
        "precipitation_visibility": "mor_visibility",
        "reflectivity": "reflectivity",
        "measurement_quality": "quality_index",
        # Weather code
        "present_weather_1m": "weather_code_synop_4680",
        "present_weather_5m": "weather_code_synop_4680_5min",
        # 'hydrometeor_type_1m' # Pickering et al., 2019
        # 'hydrometeor_type_5m' # Pickering et al., 2019
        "max_hail_diameter": "max_hail_diameter",
        "particle_count": "number_particles",
        # # Time  # to decode, lon,lat, altitude to metadata !
        # 'year'
        # 'month'
        # 'day'
        # 'hour'
        # 'minute'
        # 'second'
        # 'day_of_year'
        # Arrays
        "size_velocity_distribution": "raw_spectrum",
        "drop_size_distribution": "FieldD",
        "drop_velocity_distribution": "FieldN",
    }
    return d


def get_ARM_LPM_dict():
    # 'base_time'
    # 'time_offset'
    # 'time_bounds'
    # 'particle_diameter_bounds'
    # 'particle_fall_velocity_bounds' ???
    d = {
        "synop_4677_5min_weather_code": "weather_code_synop_4677_5min",
        "metar_4678_5min_weather_code": "weather_code_metar_4678_5min",
        "synop_4680_5min_weather_code": "weather_code_synop_4680_5min",
        "intensity_total_5min": "precipitation_rate_5min",
        "synop_4677_weather_code": "weather_code_synop_4677",
        "metar_4678_weather_code": "weather_code_metar_4678_5min",
        "synop_4680_weather_code": "weather_code_synop_4680_5min",
        "intensity_total": "precipitation_rate",
        "intensity_liquid": "rainfall_rate",
        "intensity_solid": "snowfall_rate",
        "accum_precip": "precipitation_accumulated",
        "maximum_visibility": "mor_visibility",
        "radar_reflectivity": "radar_reflectivity",
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
        "control_voltage": "voltage_control",
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
        "precipitation_spectrum": "raw_spectrum",
    }
    return d


def get_dtype_standards_all_object(sensor_name):
    # TODO: move to dev_tools I would say... is not used by any parser right?
    dtype_dict = get_L0_dtype_standards(sensor_name=sensor_name)
    for i in dtype_dict:
        dtype_dict[i] = "object"

    return dtype_dict
