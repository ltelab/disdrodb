#!/usr/bin/env python3

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2023 DISDRODB developers
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
"""Check data standards."""


import logging

import numpy as np
import pandas as pd

from disdrodb.l0.standards import (
    allowed_l0_variables,
    get_data_range_dict,
    get_valid_values_dict,
)

# Logger
from disdrodb.utils.logger import log_info

logger = logging.getLogger(__name__)


def _check_valid_range(df, dict_data_range):
    """Check valid value range of dataframe columns.

    It assumes the ``dict_data_range`` values are list ``[min_val, max_val]``.
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
        raise ValueError(msg)


def _check_valid_values(df, dict_valid_values):
    """Check valid values of dataframe columns.

    It assumes the ``dict_valid_values`` values are lists.
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
        raise ValueError(f"Columns {list_wrong_columns} have invalid values.")


def _check_raw_fields_available(
    df: pd.DataFrame,
    sensor_name: str,  # noqa: ARG001
    logger=None,  # noqa: ARG001
    verbose: bool = False,  # noqa: ARG001
) -> None:
    """Check the presence of the raw spectrum data according to the type of sensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        Error if the ``raw_drop_number`` field is missing.
    """
    # from disdrodb.l0.standards import get_raw_array_nvalues

    # Check that raw_drop_number is present
    if "raw_drop_number" not in df.columns:
        msg = "The 'raw_drop_number' column is not present in the dataframe."
        raise ValueError(msg)

    # Retrieve raw arrays that could be available (based on sensor_name)
    #  n_bins_dict = get_raw_array_nvalues(sensor_name=sensor_name)

    # Report additional raw arrays that are missing
    # raw_vars = np.array(list(n_bins_dict.keys()))
    # missing_vars = raw_vars[np.isin(raw_vars, list(df.columns), invert=True)]
    # if len(missing_vars) > 0:
    #     msg = f"The following raw array variable are missing: {missing_vars}"
    #     log_info(logger=logger, msg=msg, verbose=verbose)


def check_l0a_column_names(df: pd.DataFrame, sensor_name: str) -> None:
    """Checks that the dataframe columns respects DISDRODB standards.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        Error if some columns do not meet the DISDRODB standards or if the ``'time'``
        column is missing in the dataframe.

    """
    # Get valid columns
    valid_columns = set(allowed_l0_variables(sensor_name))

    # Get dataframe column names
    df_columns = set(df.columns)

    # Find any columns in df that aren't in the valid list
    invalid_columns = df_columns - valid_columns

    # Raise error in case
    if len(invalid_columns) > 0:
        msg = f"The following columns do no met the DISDRODB standards: {invalid_columns}"
        raise ValueError(msg)

    # Check time column is present
    if "time" not in df_columns:
        msg = "The 'time' column is missing in the dataframe."
        raise ValueError(msg)


def check_l0a_standards(df: pd.DataFrame, sensor_name: str, logger=None, verbose: bool = True) -> None:
    """Checks that a file respects the DISDRODB L0A standards.

    Parameters
    ----------
    df : pandas.DataFrame
        L0A dataframe.
    sensor_name : str
        Name of the sensor.
    verbose : bool, optional
        Whether to verbose the processing.
        The default value is ``True``.

    Raises
    ------
    ValueError
        Error if some columns have inconsistent values.

    """
    # -------------------------------------
    # Check data range
    dict_data_range = get_data_range_dict(sensor_name)
    _check_valid_range(df=df, dict_data_range=dict_data_range)

    # -------------------------------------
    # Check categorical data values
    dict_valid_values = get_valid_values_dict(sensor_name)
    _check_valid_values(df=df, dict_valid_values=dict_valid_values)

    # -------------------------------------
    # Check if raw spectrum and 1D derivate exists
    _check_raw_fields_available(df=df, sensor_name=sensor_name, logger=logger, verbose=verbose)

    # -------------------------------------
    # Check if latitude and longitude are columns of the dataframe
    # - They should be only provided if the instrument is moving !!!!
    # - TODO: this should be removed and raise error if not platform_type: mobile
    if "latitude" in df.columns:
        msg = (
            "The L0A dataframe has column 'latitude'. "
            + "This should be included only if the sensor is moving. "
            + "Otherwise, specify the 'latitude' in the metadata !"
        )
        log_info(logger=logger, msg=msg, verbose=verbose)

    if "longitude" in df.columns:
        msg = (
            "The L0A dataframe has column 'longitude'. "
            + "This should be included only if the sensor is moving. "
            + "Otherwise, specify the 'longitude' in the metadata !"
        )
        log_info(logger=logger, msg=msg, verbose=verbose)

    # -------------------------------------


def check_l0b_standards(x: str) -> None:
    """Check L0B standards."""
    # - Check for realistic values after having removed the flags !!!!
    x = "noqa"  # noqa: F841
    pass
