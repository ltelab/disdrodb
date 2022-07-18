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
import numpy as np
import pandas as pd
import dask.dataframe as dd
from disdrodb.L0.standards import get_data_format_dict, get_L0A_dtype


logger = logging.getLogger(__name__)


def available_sensor_name():
    from disdrodb.L0.standards import get_available_sensor_name
    sensor_list = get_available_sensor_name()
    raise ValueError("This need to be deprecated in favour of get_available_sensor_name() !") # TODO !!!
    return sensor_list


def check_sensor_name(sensor_name):
    from disdrodb.L0.standards import get_available_sensor_name
    available_sensor_name =  get_available_sensor_name()
    if not isinstance(sensor_name, str):
        logger.exception("'sensor_name' must be a string'")
        raise TypeError("'sensor_name' must be a string'")
    if sensor_name not in available_sensor_name:
        msg = f"Valid sensor_name are {available_sensor_name}"
        logger.exception(msg)
        raise ValueError(msg)
    return

def check_L0A_column_names(df, sensor_name):
    "Checks that the dataframe columns respects DISDRODB standards."
    # Get valid columns 
    dtype_dict = get_L0A_dtype(sensor_name)
    valid_columns = list(dtype_dict)
    valid_columns = valid_columns + ['time']
    valid_columns = set(valid_columns)  
    # Get dataframe column names
    df_columns = list(df.columns)
    df_columns = set(df_columns)
    # --------------------------------------------
    # Check there aren't valid columns 
    unvalid_columns = list(df_columns.difference(valid_columns))
    if len(unvalid_columns) > 0:
        msg = f"The following columns do no met the DISDRODB standards: {unvalid_columns}"
        logger.error(msg) 
        raise ValueError(msg)
    # --------------------------------------------
    # Check time column is present 
    if 'time' not in df_columns:
        msg = "The 'time' column is missing in the dataframe."
        logger.error(msg) 
        raise ValueError(msg)
    # --------------------------------------------
    return None


def check_array_lengths_consistency(df, sensor_name, lazy=True, verbose=False):
    from disdrodb.L0.standards import get_raw_field_nbins

    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
    list_unvalid_row_idx = []
    for key, n_bins in n_bins_dict.items():
        # Check key is available in dataframe
        if key not in df.columns:
            continue
        # Parse the string splitting at ,
        df_series = df[key].astype(str).str.split(",")
        # Check all arrays have same length
        if lazy:
            arr_lengths = df_series.apply(len, meta=(key, "int64"))
            arr_lengths = arr_lengths.compute()
        else:
            arr_lengths = df_series.apply(len)
        idx, count = np.unique(arr_lengths, return_counts=True)
        n_max_vals = idx[np.argmax(count)]
        # Idenfity rows with unexpected array length
        unvalid_row_idx = np.where(arr_lengths != n_max_vals)[0]
        if len(unvalid_row_idx) > 0:
            list_unvalid_row_idx.append(unvalid_row_idx)
    # Drop unvalid rows
    unvalid_row_idx = np.unique(list_unvalid_row_idx)
    if len(unvalid_row_idx) > 0:
        if lazy:
            n_partitions = df.npartitions
            df = df.compute()
            df = df.drop(df.index[unvalid_row_idx])
            df = dd.from_pandas(df, npartitions=n_partitions)
        else:
            df = df.drop(df.index[unvalid_row_idx])
    return df


def check_L0A_standards(fpath, sensor_name, raise_errors=False, verbose=True):
    # Read parquet
    df = pd.read_parquet(fpath)
    # -------------------------------------
    # Check data range
    dict_field_value_range = get_field_value_range_dict(sensor_name)
    list_wrong_columns = []
    for column in df.columns:
        if column in list(dict_field_value_range.keys()):
            if dict_field_value_range[column] is not None: 
                if not df[column].between(*dict_field_value_range[column]).all():
                    list_wrong_columns.append(column)
                    if raise_errors:
                        raise ValueError(f"'column' {column} has values outside the expected data range.")

    if verbose:
        if len(list_wrong_columns) > 0:
            print(" - This columns have values outside the expected data range:", list_wrong_columns)
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
        msg = " - The L0A dataframe has column 'latitude'. "
        "This should be included only if the sensor is moving. "
        "Otherwise, specify the 'latitude' in the metadata !"
        print(msg)
        logger.info(msg)

    if "longitude" in df.columns:
        msg = " - The L0A dataframe has column 'longitude'. "
        "This should be included only if the sensor is moving. "
        "Otherwise, specify the 'longitude' in the metadata !"
        print(msg)
        logger.info(msg)

    # -------------------------------------
    # Check if raw spectrum and 1D derivate exists
    list_sprectrum_vars = ["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number"]
    unavailable_vars = np.array(list_sprectrum_vars)[
        np.isin(list_sprectrum_vars, df.columns, invert=True)
    ]
    # Also if Thies_LPM has list_sprectrum_vars?
    if len(unavailable_vars) > 0:
        msg = f" - The variables {unavailable_vars} are not present in the L0 dataframe."
        print(msg)
        logger.info(msg)

    # -------------------------------------
    # Check consistency of array lengths
    # TODO
    # df = check_array_lengths_consistency(df, sensor_name, lazy=True, verbose=verbose)

    # -------------------------------------
    # Add index to dataframe
    # TODO

    # -------------------------------------
    # TODO[GG]:
    # if not respect standards, print errors and remove file (before raising errors)
    # - log, verbose ... L0A conforms to DISDRODB standards ;)

    # -------------------------------------
    return


def check_L0B_standards(x):
    # TODO:
    # - Check for realistic values after having removed the flags !!!!
    pass


####--------------------------------------------------------------------------.
#### Get instrument default string standards
def get_field_ndigits_natural_dict(sensor_name):
    """Get number of digits on th left side of the comma."""
    # (example: 123,45 -> 123)
    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_naturals"] for k, v in data_dict.items()}
    return d


def get_field_ndigits_decimals_dict(sensor_name):
    """Get number of digits on the right side of the comma."""
    # (example: 123,45 -> 45)
    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_decimals"] for k, v in data_dict.items()}
    return d


def get_field_ndigits_dict(sensor_name):
    """Get number of digits

    It excludes the comma but it count the minus sign !!!.
    """
    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_digits"] for k, v in data_dict.items()}
    return d


def get_field_nchar_dict(sensor_name):
    """Get the total number of characters.

    It accounts also for the comma and the minus sign.
    """
    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_characters"] for k, v in data_dict.items()}
    return d


def get_field_value_range_dict(sensor_name):
    """Get the variable data range (including nan flags)."""
    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["data_range"] for k, v in data_dict.items()}
    return d


def get_field_flag_dict(sensor_name):
    """Get the variable nan flags."""
    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["nan_flags"] for k, v in data_dict.items()}
    return d


# TODO: get_field_value_realistic_range  # when removing flags

####--------------------------------------------------------------------------.
#### Instrument default numeric standards
# TODO: this is currently used?
# --> YAML file with a list of variables providing error infos?
# --> And provide values that represents errors?


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
    elif sensor_name == "Thies_LPM":
        value_dict = {
            "laser_status": [0, 1],
            "laser_temperature_analog_status": [0, 1],
            "laser_temperature_digital_status": [0, 1],
            "laser_current_analog_status": [0, 1],
            "laser_current_digital_status": [0, 1],
            "sensor_voltage_supply_status": [0, 1],
            "current_heating_pane_transmitter_head_status": [0, 1],
            "current_heating_pane_receiver_head_status": [0, 1],
            "temperature_sensor_status": [0, 1],
            "current_heating_voltage_supply_status": [0, 1],
            "current_heating_house_status": [0, 1],
            "current_heating_heads_status": [0, 1],
            "current_heating_carriers_status": [0, 1],
            "control_output_laser_power_status": [0, 1],
            "reserve_status": [0, 1]
            # TODO: weather codes
            # Faculative/custom fields
        }
    else:
        raise NotImplementedError

    return value_dict


def get_field_error_dict(device):
    if device == "OTT_Parsivel":
        flag_dict = {
            "sensor_status": [1, 2, 3],
            "datalogger_error": [1],
            "error_code": [1, 2],
        }
    return flag_dict
