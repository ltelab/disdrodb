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
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def available_sensor_name():
    from disdrodb.standards import get_available_sensor_name

    sensor_list = get_available_sensor_name()
    return sensor_list


def check_sensor_name(sensor_name):
    if not isinstance(sensor_name, str):
        logger.exception("'sensor_name' must be a string'")
        raise TypeError("'sensor_name' must be a string'")
    if sensor_name not in available_sensor_name():
        msg = f"Valid sensor_name are {available_sensor_name()}"
        logger.exception(msg)
        raise ValueError(msg)
    return


def check_L0_column_names(x):
    # Allow TO_BE_SPLITTED, TO_BE_PARSED
    pass


def check_L0_standards(fpath, sensor_name, raise_errors=False, verbose=True):
    # Read parquet
    df = pd.read_parquet(fpath)
    # -------------------------------------
    # Check data range
    dict_field_value_range = get_field_value_range_dict(sensor_name)
    list_wrong_columns = []
    for column in df.columns:
        if column in list(dict_field_value_range.keys()):
            if not df[column].between(*dict_field_value_range[column]).all():
                list_wrong_columns.append(column)
                if raise_errors:
                    raise ValueError(
                        f"'column' {column} has values outside the expected data range."
                    )
    if verbose:
        if len(list_wrong_columns) > 0:
            print(
                " - The following columns have values outside the expected data range:",
                list_wrong_columns,
            )
    # -------------------------------------
    # Check categorical data values
    dict_field_values = get_field_value_options_dict(sensor_name)
    list_wrong_columns = []
    list_msg = []
    for column in df.columns:
        if column in list(dict_field_values.keys()):
            if not df[column].isin(dict_field_values[column]).all():
                list_wrong_columns.append(column)
                if raise_errors:
                    msg = f"'column' {column} has values different from {dict_field_values[column]}"
                    list_msg.append(msg)
                    raise ValueError(msg)
    if verbose:
        if len(list_wrong_columns) > 0:
            print(
                " - The following columns have values outside the expected data range:",
                list_wrong_columns,
            )
            [print(msg) for msg in list_msg]
    # -------------------------------------
    # Check if latitude and longitude are columns of the dataframe
    # - They should be only provided if the instrument is moving !!!!
    if "latitude" in df.columns:
        msg = " - The L0 dataframe has column 'latitude'. "
        "This should be included only if the sensor is moving. "
        "Otherwise, specify the 'latitude' in the metadata !"
        print(msg)
        logger.info(msg)

    if "longitude" in df.columns:
        msg = " - The L0 dataframe has column 'longitude'. "
        "This should be included only if the sensor is moving. "
        "Otherwise, specify the 'longitude' in the metadata !"
        print(msg)
        logger.info(msg)

    # -------------------------------------
    # Check consistency of array lengths
    # TODO

    # -------------------------------------
    # Add index to dataframe
    # TODO

    # -------------------------------------
    # TODO[GG]:
    # if not respect standards, print errors and remove file (before raising errors)
    # - log, verbose ... L0 conforms to DISDRODB standards ;)

    # -------------------------------------
    return


def check_L1_standards(x):
    # TODO:
    # - Check for realistic values after having removed the flags !!!!
    pass


def check_L2_standards(x):
    # TODO:
    pass


####--------------------------------------------------------------------------.
#### Instrument default string standards
# Numbers of digit on the left side of comma (example: 123,45 -> 123)
def get_field_ndigits_natural_dict(sensor_name):

    if sensor_name == "OTT_Parsivel":
        digits_dict = {
            # Mandatory
            "rain_rate_16bit": 4,
            "rain_rate_32bit": 4,
            "rain_accumulated_16bit": 4,
            "rain_accumulated_32bit": 4,
            "rain_amount_absolute_32bit": 3,
            "reflectivity_16bit": 2,
            "reflectivity_32bit": 2,
            "rain_kinetic_energy": 3,
            "snowfall_intensity": 4,
            "mor_visibility": 4,
            "weather_code_SYNOP_4680": 2,
            "weather_code_SYNOP_4677": 2,
            #
            #
            "n_particles": 5,
            "sensor_temperature": 3,
            "sensor_heating_current": 1,
            "sensor_battery_voltage": 2,
            "sensor_status": 1,
            "laser_amplitude": 5,
            "error_code": 3,
            "FieldN": 0,
            "FieldV": 0,
            "RawData": 0,
            # Faculative/custom fields
            "datalogger_temperature": 0,
            "datalogger_voltage": 0,
            "datalogger_error": 1,
        }
    elif sensor_name == "OTT_Parsivel2":
        digits_dict = {
            # Mandatory
            "rain_rate_16bit": 4,
            "rain_rate_32bit": 4,
            "rain_accumulated_16bit": 4,
            "rain_accumulated_32bit": 4,
            "rain_amount_absolute_32bit": 3,
            "reflectivity_16bit": 2,
            "reflectivity_32bit": 2,
            "rain_kinetic_energy": 3,
            "snowfall_intensity": 4,
            "mor_visibility": 5,
            "weather_code_SYNOP_4680": 2,
            "weather_code_SYNOP_4677": 2,
            "n_particles": 5,
            "n_particles_all": 2,
            "sensor_temperature": 3,
            "temperature_PBC": 2,
            "temperature_right": 2,
            "temperature_left": 2,
            "sensor_heating_current": 1,
            "sensor_battery_voltage": 2,
            "sensor_status": 1,
            "laser_amplitude": 5,
            "error_code": 3,
            "FieldN": 0,
            "FieldV": 0,
            "RawData": 0,
            # Faculative/custom fields
            "datalogger_temperature": 0,
            "datalogger_voltage": 0,
            "datalogger_error": 1,
        }

    else:
        raise NotImplementedError

    return digits_dict


# Numbers of digit on the right side of comma (example: 123,45 -> 45)
def get_field_ndigits_decimals_dict(sensor_name):

    if sensor_name == "OTT_Parsivel":
        digits_dict = {
            # Mandatory
            "rain_rate_16bit": 4,
            "rain_rate_32bit": 3,
            "rain_accumulated_16bit": 2,
            "rain_accumulated_32bit": 2,
            "rain_amount_absolute_32bit": 3,
            "reflectivity_16bit": 2,
            "reflectivity_32bit": 3,
            "rain_kinetic_energy": 3,
            "snowfall_intensity": 2,
            "mor_visibility": 0,
            "weather_code_SYNOP_4680": 0,
            "weather_code_SYNOP_4677": 0,
            "n_particles": 0,
            "sensor_temperature": 0,
            "sensor_heating_current": 2,
            "sensor_battery_voltage": 1,
            "sensor_status": 0,
            "laser_amplitude": 0,
            "error_code": 0,
            "FieldN": 0,
            "FieldV": 0,
            "RawData": 0,
            # Faculative/custom fields
            "datalogger_temperature": 0,
            "datalogger_voltage": 0,
            "datalogger_error": 0,
        }
    elif sensor_name == "OTT_Parsivel2":
        digits_dict = {
            # Mandatory
            "rain_rate_16bit": 4,
            "rain_rate_32bit": 3,
            "rain_accumulated_16bit": 2,
            "rain_accumulated_32bit": 2,
            "rain_amount_absolute_32bit": 3,
            "reflectivity_16bit": 2,
            "reflectivity_32bit": 3,
            "rain_kinetic_energy": 3,
            "snowfall_intensity": 2,
            "mor_visibility": 0,
            "weather_code_SYNOP_4680": 0,
            "weather_code_SYNOP_4677": 0,
            "n_particles": 0,
            "n_particles_all": 0,
            "sensor_temperature": 0,
            "temperature_PBC": 0,
            "temperature_right": 0,
            "temperature_left": 0,
            "sensor_heating_current": 2,
            "sensor_battery_voltage": 1,
            "sensor_status": 0,
            "laser_amplitude": 0,
            "error_code": 0,  # Not in manual
            "FieldN": 0,
            "FieldV": 0,
            "RawData": 0,
            # Faculative/custom fields
            "datalogger_temperature": 0,
            "datalogger_voltage": 0,
            "datalogger_error": 0,
        }

    else:
        raise NotImplementedError

    return digits_dict


def get_field_ndigits_dict(sensor_name):
    # Do not count .
    # Do count minus sign -
    if sensor_name == "OTT_Parsivel":
        digits_dict = {
            # Mandatory
            "rain_rate_16bit": 8,
            "rain_rate_32bit": 7,
            "rain_accumulated_16bit": 7,
            "rain_accumulated_32bit": 6,
            "rain_amount_absolute_32bit": 6,
            "reflectivity_16bit": 4,
            "reflectivity_32bit": 5,
            "rain_kinetic_energy": 7,
            "snowfall_intensity": 7,
            "mor_visibility": 4,
            "weather_code_SYNOP_4680": 2,
            "weather_code_SYNOP_4677": 2,
            "n_particles": 5,
            "sensor_temperature": 3,
            "sensor_heating_current": 3,
            "sensor_battery_voltage": 3,
            "sensor_status": 1,
            "laser_amplitude": 5,
            "error_code": 3,
            "FieldN": 0,
            "FieldV": 0,
            "RawData": 0,
            # Faculative/custom fields
            "datalogger_temperature": 3,
            "datalogger_voltage": 0,
            "datalogger_error": 1,
        }
    elif sensor_name == "OTT_Parsivel2":
        digits_dict = {
            # Mandatory
            "rain_rate_16bit": 8,
            "rain_rate_32bit": 8,
            "rain_accumulated_16bit": 7,
            "rain_accumulated_32bit": 6,
            "rain_amount_absolute_32bit": 6,
            "reflectivity_16bit": 4,
            "reflectivity_32bit": 5,
            "rain_kinetic_energy": 7,
            "snowfall_intensity": 7,
            "mor_visibility": 4,
            "weather_code_SYNOP_4680": 2,
            "weather_code_SYNOP_4677": 2,
            "n_particles": 5,
            "n_particles_all": 8,
            "sensor_temperature": 3,
            "temperature_PBC": 3,
            "temperature_right": 3,
            "temperature_left": 3,
            "sensor_heating_current": 3,
            "sensor_battery_voltage": 3,
            "sensor_status": 1,
            "laser_amplitude": 5,
            "error_code": 3,
            "FieldN": 0,
            "FieldV": 0,
            "RawData": 0,
            # Faculative/custom fields
            "datalogger_temperature": 3,
            "datalogger_voltage": 0,
            "datalogger_error": 1,
        }
    else:
        raise NotImplementedError

    return digits_dict


# Numbers of all digit (example: 123,45 -> 123,45)
def get_field_nchar_dict(sensor_name):
    ### Count also .
    if sensor_name == "OTT_Parsivel":
        digits_dict = {
            # Mandatory
            "rain_rate_16bit": 5,
            "rain_rate_32bit": 8,
            "rain_accumulated_16bit": 6,
            "rain_accumulated_32bit": 7,
            "rain_amount_absolute_32bit": 7,
            "reflectivity_16bit": 5,
            "reflectivity_32bit": 6,
            "rain_kinetic_energy": 6,
            "snowfall_intensity": 6,
            "mor_visibility": 4,
            "weather_code_SYNOP_4680": 2,
            "weather_code_SYNOP_4677": 2,
            "n_particles": 5,
            "sensor_temperature": 3,
            "sensor_heating_current": 4,
            "sensor_battery_voltage": 4,
            "sensor_status": 1,
            "laser_amplitude": 5,
            "error_code": 3,
            "FieldN": 224,
            "FieldV": 224,
            "RawData": 4096,
            # Faculative/custom fields
            "datalogger_temperature": 3,
            "datalogger_voltage": 2,
            "datalogger_error": 1,
        }
    elif sensor_name == "OTT_Parsivel2":

        digits_dict = {
            # Mandatory
            "rain_rate_16bit": 5,
            "rain_rate_32bit": 7,
            "rain_accumulated_16bit": 6,
            "rain_accumulated_32bit": 7,
            "rain_amount_absolute_32bit": 7,
            "reflectivity_16bit": 5,
            "reflectivity_32bit": 6,
            "rain_kinetic_energy": 6,
            "snowfall_intensity": 6,
            "mor_visibility": 4,
            "weather_code_SYNOP_4680": 2,
            "weather_code_SYNOP_4677": 2,
            "n_particles": 5,
            "n_particles_all": 8,
            "sensor_temperature": 3,
            "temperature_PBC": 3,
            "temperature_right": 3,
            "temperature_left": 3,
            "sensor_heating_current": 4,
            "sensor_battery_voltage": 4,
            "sensor_status": 1,
            "laser_amplitude": 5,
            "error_code": 3,
            "FieldN": 224,
            "FieldV": 224,
            "RawData": 4096,
            # Faculative/custom fields
            "datalogger_temperature": 3,
            "datalogger_voltage": 2,
            "datalogger_error": 1,
        }
    else:
        raise NotImplementedError

    return digits_dict


####--------------------------------------------------------------------------.
#### Instrument default numeric standards
# TODO: get_field_flag_values
# TODO: get_field_value_realistic_range  # when removing flags


def get_field_value_options_dict(sensor_name):
    if sensor_name == "OTT_Parsivel":
        value_dict = {
            "sensor_status": [0, 1, 2, 3],
            "error_code": [0, 1, 2],
            # TODO: weather codes
            # Faculative/custom fields
            # 'datalogger_temperature': ['NaN']
            "datalogger_voltage": ["OK", "KO"],
            "datalogger_error": [0],
        }
    elif sensor_name == "OTT_Parsivel2":
        value_dict = {
            "sensor_status": [0, 1, 2, 3],
            "error_code": [0, 1, 2],
            # TODO: weather codes
            # Faculative/custom fields
        }
    else:
        raise NotImplementedError

    return value_dict


def get_field_value_range_dict(sensor_name):
    if sensor_name == "OTT_Parsivel":
        range_dict = {
            # Mandatory
            "rain_rate_16bit": [0, 9999.999],
            "rain_rate_32bit": [0, 9999.999],
            "rain_accumulated_16bit": [0, 300.00],
            "rain_accumulated_32bit": [0, 300.00],
            "rain_amount_absolute_32bit": [0, 999.999],
            "reflectivity_16bit": [-9.999, 99.999],
            "reflectivity_32bit": [-9.999, 99.999],
            "rain_kinetic_energy": [0, 999.999],
            "snowfall_intensity": [0, 999.999],
            "mor_visibility": [0, 20000],
            "weather_code_SYNOP_4680": [0, 99],
            "weather_code_SYNOP_4677": [0, 99],
            "n_particles": [0, 99999],  # For debug, [0, 99999]
            "sensor_temperature": [-99, 100],
            "sensor_heating_current": [0, 4.00],
            "sensor_battery_voltage": [0, 30.0],
            "sensor_status": [0, 3],
            "laser_amplitude": [0, 99999],
            "error_code": [0, 3],
            # Faculative/custom fields
            "latitude": [-90, 90],
            "longitude": [-180, 180],
        }
    elif sensor_name == "OTT_Parsivel2":
        range_dict = {
            # Mandatory
            "rain_rate_16bit": [0, 9999.999],
            "rain_rate_32bit": [0, 9999.999],
            "rain_accumulated_16bit": [0, 300.00],
            "rain_accumulated_32bit": [0, 300.00],
            "rain_amount_absolute_32bit": [0, 999.999],
            "reflectivity_16bit": [-9.999, 99.999],
            "reflectivity_32bit": [-9.999, 99.999],
            "rain_kinetic_energy": [0, 999.999],
            "snowfall_intensity": [0, 999.999],
            "mor_visibility": [0, 20000],
            "weather_code_SYNOP_4680": [0, 99],
            "weather_code_SYNOP_4677": [0, 99],
            "n_particles": [0, 99999],  # For debug, [0, 99999]
            "n_particles_all": [0, 8192],
            "sensor_temperature": [-99, 100],
            "temperature_PBC": [-99, 100],
            "temperature_right": [-99, 100],
            "temperature_left": [-99, 100],
            "sensor_heating_current": [0, 4.00],
            "sensor_battery_voltage": [0, 30.0],
            "sensor_status": [0, 3],
            "laser_amplitude": [0, 99999],
            "error_code": [0, 3],
            # Faculative/custom fields
            "latitude": [-90, 90],
            "longitude": [-180, 180],
        }
    else:
        raise NotImplementedError

    return range_dict


####--------------------------------------------------------------------------.
def get_field_error_dict(device):
    if device == "OTT_Parsivel":
        flag_dict = {
            "sensor_status": [1, 2, 3],
            "datalogger_error": [1],
            "error_code": [1, 2],
        }
    return flag_dict
