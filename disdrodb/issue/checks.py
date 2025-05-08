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
"""Checks for issue YAML files."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _is_numpy_array_string(arr):
    """Check if the numpy array contains strings.

    Parameters
    ----------
    arr : numpy array
        Numpy array to check.
    """
    return np.issubdtype(arr.dtype, np.str_)


def _is_numpy_array_datetime(arr):
    """Check if the numpy array contains datetime64.

    Parameters
    ----------
    arr : numpy array
        Numpy array to check.

    Returns
    -------
    numpy array
        Numpy array checked.
    """
    return np.issubdtype(arr.dtype, np.datetime64)


def _check_timestep_datetime_accuracy(timesteps, unit="s"):
    """Check the accuracy of the numpy datetime array.

    Parameters
    ----------
    timesteps : numpy array
        Numpy array to check.
    unit : str, optional
        Time unit. The default value is "s".

    Returns
    -------
    numpy array
        Numpy array checked.

    Raises
    ------
    ValueError
    """
    if timesteps.dtype != f"<M8[{unit}]":
        msg = f"The timesteps does not have datetime64 {unit} accuracy."
        raise ValueError(msg)
    return timesteps


def _check_timestep_string_second_accuracy(timesteps):
    """Check the timesteps string are provided with second accuracy.

    Note: it assumes the YYYY-mm-dd HH:MM:SS format
    """
    n_characters = np.char.str_len(timesteps)
    mispecified_timesteps = timesteps[n_characters != 19]
    if len(mispecified_timesteps) > 0:
        msg = (
            f"The following timesteps are mispecified: {mispecified_timesteps}. Expecting the YYYY-mm-dd HH:MM:SS"
            " format."
        )
        raise ValueError(msg)
    return timesteps


def _check_timesteps_string(timesteps):
    """Check timesteps string validity.

    It expects a list of timesteps strings in YYYY-mm-dd HH:MM:SS format with second accuracy.
    """
    timesteps = np.asarray(timesteps)
    timesteps = _check_timestep_string_second_accuracy(timesteps)
    # Reformat as datetime64 with second accuracy
    new_timesteps = pd.to_datetime(timesteps, format="%Y-%m-%d %H:%M:%S", errors="coerce").astype("M8[s]")
    # Raise errors if timesteps are mispecified
    idx_mispecified = np.isnan(new_timesteps)
    mispecified_timesteps = timesteps[idx_mispecified].tolist()
    if len(mispecified_timesteps) > 0:
        msg = (
            f"The following timesteps are mispecified: {mispecified_timesteps}. Expecting the YYYY-mm-dd HH:MM:SS"
            " format."
        )
        raise ValueError(msg)
    # Convert to numpy
    new_timesteps = new_timesteps.to_numpy()
    return new_timesteps


def check_timesteps(timesteps):
    """Check timesteps validity.

    It expects timesteps string in YYYY-mm-dd HH:MM:SS format with second accuracy.
    If timesteps is None, return None.
    """
    if isinstance(timesteps, type(None)):
        return None
    if isinstance(timesteps, str):
        timesteps = [timesteps]
    # Set as numpy array
    timesteps = np.array(timesteps)
    # If strings, check accordingly
    if _is_numpy_array_string(timesteps):
        timesteps = _check_timesteps_string(timesteps)
    # If numpy datetime64, check accordingly
    elif _is_numpy_array_datetime(timesteps):
        timesteps = _check_timestep_datetime_accuracy(timesteps, unit="s")
    else:
        raise TypeError("Invalid timesteps input.")
    return timesteps


def _check_time_period_nested_list_format(time_periods):
    """Check that the time_periods is a list of list of length 2."""
    if not isinstance(time_periods, list):
        msg = "'time_periods' must be a list'"
        raise TypeError(msg)

    for time_period in time_periods:
        if not isinstance(time_period, (list, np.ndarray)) or len(time_period) != 2:
            msg = "Every time period of time_periods must be a list of length 2."
            raise ValueError(msg)


def check_time_periods(time_periods):
    """Check time_periods validity."""
    # Return None if None
    if isinstance(time_periods, type(None)):
        return None
    # Check time_period format
    _check_time_period_nested_list_format(time_periods)
    # Convert each time period to datetime64
    new_time_periods = []
    for time_period in time_periods:
        time_period = check_timesteps(timesteps=time_period)
        new_time_periods.append(time_period)
    # Check time period start occur before end
    for time_period in new_time_periods:
        if time_period[0] > time_period[1]:
            msg = f"The {time_period} time_period is invalid. Start time occurs after end time."
            raise ValueError(msg)
    return new_time_periods


def _get_issue_timesteps(issue_dict):
    """Get timesteps from issue dictionary."""
    timesteps = issue_dict.get("timesteps", None)
    # Check validity
    timesteps = check_timesteps(timesteps)
    # Sort
    timesteps.sort()
    return timesteps


def _get_issue_time_periods(issue_dict):
    """Get time_periods from issue dictionary."""
    time_periods = issue_dict.get("time_periods", None)
    time_periods = check_time_periods(time_periods)
    return time_periods


def check_issue_dict(issue_dict):
    """Check validity of the issue dictionary."""
    # Check is empty
    if len(issue_dict) == 0:
        return issue_dict
    # Check there are only timesteps and time_periods keys
    valid_keys = ["timesteps", "time_periods"]
    keys = list(issue_dict.keys())
    invalid_keys = [k for k in keys if k not in valid_keys]
    if len(invalid_keys) > 0:
        msg = f"Invalid {invalid_keys} keys. The issue YAML file accept only {valid_keys}"
        raise ValueError(msg)

    # Check timesteps
    timesteps = _get_issue_timesteps(issue_dict)
    # Check time periods
    time_periods = _get_issue_time_periods(issue_dict)
    # Recreate issue dict
    issue_dict["timesteps"] = timesteps
    issue_dict["time_periods"] = time_periods

    return issue_dict


def check_issue_compliance(data_source, campaign_name, station_name, metadata_archive_dir=None):
    """Check DISDRODB issue compliance."""
    from disdrodb.issue.reader import read_station_issue

    issue_dict = read_station_issue(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    issue_dict = check_issue_dict(issue_dict)
