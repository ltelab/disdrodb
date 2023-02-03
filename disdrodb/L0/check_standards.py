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
from typing import Union
from disdrodb.L0.standards import get_data_format_dict, get_L0A_dtype

# Logger
from disdrodb.utils.logger import (
    log_info,
    log_warning,
    log_error,
    log_debug,
)

logger = logging.getLogger(__name__)


def _check_valid_range(df, dict_value_range, verbose=False):
    """Check valid value range of dataframe columns."""
    list_wrong_columns = []
    for column in df.columns:
        if column in list(dict_value_range.keys()):
            if dict_value_range[column] is not None:
                if not df[column].between(*dict_value_range[column]).all():
                    list_wrong_columns.append(column)

    if len(list_wrong_columns) > 0:
        msg = (
            f"Columns {list_wrong_columns} has values outside the expected data range."
        )
        log_error(logger=logger, msg=msg, verbose=verbose)
        raise ValueError(msg)


def _check_valid_values(df, dict_valid_values, verbose=False):
    """Check valid values of dataframe columns."""
    list_msg = []
    list_wrong_columns = []
    for column in df.columns:
        if column in list(dict_valid_values.keys()):
            valid_values = dict_valid_values[column]
            if not df[column].isin(valid_values).all():
                list_wrong_columns.append(column)
                msg = f"The column {column} has values different from {valid_values}."
                list_msg.append(msg)

    if len(list_wrong_columns) > 0:
        msg = "\n".join(list_msg)
        log_error(logger=logger, msg=msg, verbose=verbose)
        raise ValueError(f"Columns {list_wrong_columns} have invalid values.")


def _check_raw_fields_available(
    df: pd.DataFrame, sensor_name: str, verbose: bool = False
) -> None:
    """Check the presence of the raw spectrum data according to the type of sensor.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        Error if the raw_drop_number field is missing.
    """
    from disdrodb.L0.standards import get_raw_field_nbins

    # Retrieve raw arrays that could be available (based on sensor_name)
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
    raw_vars = np.array(list(n_bins_dict.keys()))

    # Check that raw_drop_number is present
    if not "raw_drop_number" in df.columns:
        msg = "The 'raw_drop_number' column is not present in the dataframe."
        log_error(logger=logger, msg=msg, verbose=verbose)
        raise ValueError(msg)

    # Report additional raw arrays that are missing
    missing_vars = raw_vars[np.isin(raw_vars, list(df.columns), invert=True)]
    if len(missing_vars) > 0:
        msg = f"The following raw array variable aree missing: {missing_vars}"
        log_info(logger=logger, msg=msg, verbose=verbose)
    return None


def check_sensor_name(sensor_name: str) -> None:
    """Check sensor name.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    TypeError
        Error if `sensor_name` is not a string.
    ValueError
        Error if the input sensor name has not been found in the list of available sensors.
    """
    from disdrodb.L0.standards import get_available_sensor_name

    available_sensor_name = get_available_sensor_name()
    if not isinstance(sensor_name, str):
        raise TypeError("'sensor_name' must be a string.")
    if sensor_name not in available_sensor_name:
        msg = f"{sensor_name} not valid {sensor_name}. Valid values are {available_sensor_name}."
        logger.error(msg)
        raise ValueError(msg)


def check_L0A_column_names(df: pd.DataFrame, sensor_name: str) -> None:
    """Checks that the dataframe columns respects DISDRODB standards.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        Error if some columns do not meet the DISDRODB standards or if the 'time' column is missing in the dataframe.

    """

    # Get valid columns
    dtype_dict = get_L0A_dtype(sensor_name)
    valid_columns = list(dtype_dict)
    valid_columns = valid_columns + ["time"]
    valid_columns = set(valid_columns)
    # Get dataframe column names
    df_columns = list(df.columns)
    df_columns = set(df_columns)
    # --------------------------------------------
    # Check there aren't valid columns
    unvalid_columns = list(df_columns.difference(valid_columns))
    if len(unvalid_columns) > 0:
        msg = (
            f"The following columns do no met the DISDRODB standards: {unvalid_columns}"
        )
        logger.error(msg)
        raise ValueError(msg)
    # --------------------------------------------
    # Check time column is present
    if "time" not in df_columns:
        msg = "The 'time' column is missing in the dataframe."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    # --------------------------------------------
    return None


def check_L0A_standards(
    df: pd.DataFrame, sensor_name: str, verbose: bool = True
) -> None:
    """Checks that a file respects the DISDRODB L0A standards.

    Parameters
    ----------
    df : pd.DataFrame
        L0A dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool, optional
        Wheter to verbose the processing.
        The default is True.

    Raises
    ------
    ValueError
        Error if some columns have inconsistent values.

    """
    # -------------------------------------
    # Check data range
    dict_value_range = get_field_value_range_dict(sensor_name)
    _check_valid_range(df=df, dict_value_range=dict_value_range, verbose=verbose)

    # -------------------------------------
    # Check categorical data values
    dict_valid_values = get_field_value_options_dict(sensor_name)
    _check_valid_values(df=df, dict_valid_values=dict_valid_values, verbose=verbose)

    # -------------------------------------
    # Check if raw spectrum and 1D derivate exists
    _check_raw_fields_available(df=df, sensor_name=sensor_name, verbose=verbose)

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
    # Check that numeric variable are not all NaN
    # - Otherwise raise error
    # - df[column].isna().all()

    # -------------------------------------


def check_L0B_standards(x: str) -> None:
    # TODO:
    # - Check for realistic values after having removed the flags !!!!
    pass


####--------------------------------------------------------------------------.
#### Get instrument default string standards
def get_field_ndigits_natural_dict(sensor_name: str) -> dict:
    """Get number of digits on the left side of the comma from the instrument default string standards.

    Example: 123,45 -> 123 --> 3 natural digits

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected number of natural digits for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_naturals"] for k, v in data_dict.items()}
    return d


def get_field_ndigits_decimals_dict(sensor_name: dict) -> dict:
    """Get number of digits on the right side of the comma from the instrument default string standards.

    Example: 123,45 -> 45 --> 2 decimal digits
    Parameters
    ----------
    sensor_name : dict
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected number of decimal digits for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_decimals"] for k, v in data_dict.items()}
    return d


def get_field_ndigits_dict(sensor_name: str) -> dict:
    """Get number of digits from the instrument default string standards.

    Important note: it excludes the comma but it counts the minus sign !!!


    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    Returns
    -------
    dict
        Dictionary with the expected number of digits for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_digits"] for k, v in data_dict.items()}
    return d


def get_field_nchar_dict(sensor_name: str) -> dict:
    """Get the total number of characters from the instrument default string standards.

    Important note: it accounts also for the comma and the minus sign !!!


    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected number of characters for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["n_characters"] for k, v in data_dict.items()}
    return d


def get_field_value_range_dict(sensor_name: str) -> dict:
    """Get the variable data range.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected data value range for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["data_range"] for k, v in data_dict.items()}
    return d


def get_field_flag_dict(sensor_name: str) -> dict:
    """Get the variable nan flags.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with the expected nan flag for each data field.
    """

    data_dict = get_data_format_dict(sensor_name)
    d = {k: v["nan_flags"] for k, v in data_dict.items()}
    return d


####--------------------------------------------------------------------------.
#### Instrument default numeric standards
# TODO: this is currently used?
# --> YAML file with a list of variables providing error infos?
# --> And provide values that represents errors?


def get_field_value_options_dict(sensor_name: str) -> dict:
    """Get the dictionary of field values.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary of field values.

    Raises
    ------
    NotImplementedError
        Error if the name of the sensor is not available.
    """
    # TODO: this should go in data_format YAML file!
    if sensor_name == "OTT_Parsivel":
        value_dict = {
            "sensor_status": [0, 1, 2, 3],
            "error_code": [0, 1, 2],
            # TODO: weather codes
            # Faculative/custom fields
            # 'datalogger_temperature': ['NaN']
            # "datalogger_voltage": ["OK", "KO"],
            # "datalogger_error": [0],
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


def get_field_error_dict(device: str) -> dict:
    """Get field error dictionnary

    Parameters
    ----------
    device : str
        Name of the sensor

    Returns
    -------
    dict
        Dictionnary of the field error
    """
    if device == "OTT_Parsivel":
        flag_dict = {
            "sensor_status": [1, 2, 3],
            # "datalogger_error": [1],
            "error_code": [1, 2],
        }
    return flag_dict
