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
from disdrodb.l0.standards import (
    get_l0a_dtype,
    get_valid_values_dict,
    get_data_range_dict,
)

# Logger
from disdrodb.utils.logger import (
    log_info,
    log_error,
)

logger = logging.getLogger(__name__)


def _check_valid_range(df, dict_data_range, verbose=False):
    """Check valid value range of dataframe columns.

    It assumes the dict_data_range values are list [min_val, max_val]
    """
    list_wrong_columns = []
    for column in df.columns:
        if column in list(dict_data_range.keys()):
            # If nan occurs, assume it as valid values
            idx_nan = np.isnan(df[column])
            idx_valid = df[column].between(*dict_data_range[column])
            idx_valid = np.logical_or(idx_valid, idx_nan)
            if not idx_valid.all():
                list_wrong_columns.append(column)

    if len(list_wrong_columns) > 0:
        msg = f"Columns {list_wrong_columns} has values outside the expected data range."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)


def _check_valid_values(df, dict_valid_values, verbose=False):
    """Check valid values of dataframe columns.

    It assumes the dict_valid_values values are list [...].
    """
    list_msg = []
    list_wrong_columns = []
    for column in df.columns:
        if column in list(dict_valid_values.keys()):
            valid_values = dict_valid_values[column]
            # If nan occurs, assume it as valid values
            idx_nan = np.isnan(df[column])
            idx_valid = df[column].isin(valid_values)
            idx_valid = np.logical_or(idx_valid, idx_nan)
            if not idx_valid.all():
                list_wrong_columns.append(column)
                msg = f"The column {column} has values different from {valid_values}."
                list_msg.append(msg)

    if len(list_wrong_columns) > 0:
        msg = "\n".join(list_msg)
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(f"Columns {list_wrong_columns} have invalid values.")


def _check_raw_fields_available(df: pd.DataFrame, sensor_name: str, verbose: bool = False) -> None:
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
    from disdrodb.l0.standards import get_raw_array_nvalues

    # Retrieve raw arrays that could be available (based on sensor_name)
    n_bins_dict = get_raw_array_nvalues(sensor_name=sensor_name)
    raw_vars = np.array(list(n_bins_dict.keys()))

    # Check that raw_drop_number is present
    if "raw_drop_number" not in df.columns:
        msg = "The 'raw_drop_number' column is not present in the dataframe."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)

    # Report additional raw arrays that are missing
    missing_vars = raw_vars[np.isin(raw_vars, list(df.columns), invert=True)]
    if len(missing_vars) > 0:
        msg = f"The following raw array variable are missing: {missing_vars}"
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
    from disdrodb.l0.standards import available_sensor_name

    available_sensor_name = available_sensor_name()
    if not isinstance(sensor_name, str):
        raise TypeError("'sensor_name' must be a string.")
    if sensor_name not in available_sensor_name:
        msg = f"{sensor_name} not valid {sensor_name}. Valid values are {available_sensor_name}."
        logger.error(msg)
        raise ValueError(msg)


def check_l0a_column_names(df: pd.DataFrame, sensor_name: str) -> None:
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
    dtype_dict = get_l0a_dtype(sensor_name)
    valid_columns = list(dtype_dict)
    valid_columns = valid_columns + ["time", "latitude", "longitude"]
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
    if "time" not in df_columns:
        msg = "The 'time' column is missing in the dataframe."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    # --------------------------------------------
    return None


def check_l0a_standards(df: pd.DataFrame, sensor_name: str, verbose: bool = True) -> None:
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
    dict_data_range = get_data_range_dict(sensor_name)
    _check_valid_range(df=df, dict_data_range=dict_data_range, verbose=verbose)

    # -------------------------------------
    # Check categorical data values
    dict_valid_values = get_valid_values_dict(sensor_name)
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
        log_info(logger=logger, msg=msg, verbose=verbose)

    if "longitude" in df.columns:
        msg = " - The L0A dataframe has column 'longitude'. "
        "This should be included only if the sensor is moving. "
        "Otherwise, specify the 'longitude' in the metadata !"
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -------------------------------------


def check_l0b_standards(x: str) -> None:
    # TODO:
    # - Check for realistic values after having removed the flags !!!!
    pass
