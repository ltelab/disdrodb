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

import os
import yaml
import logging
import numpy as np
import pandas as pd
from disdrodb.utils.logger import log_error


logger = logging.getLogger(__name__)

####--------------------------------------------------------------------------.
#### Checks


def is_numpy_array_string(arr):
    """Check if the numpy array contains strings

    Parameters
    ----------
    arr : numpy array
        Numpy array to check.
    """

    dtype = arr.dtype.type
    return dtype == np.str_ or dtype == np.unicode_


def is_numpy_array_datetime(arr):
    """Check if the numpy array contains datetime64

    Parameters
    ----------
    arr : numpy array
        Numpy array to check.

    Returns
    -------
    numpy array
        Numpy array checked.
    """
    return arr.dtype.type == np.datetime64


def _check_timestep_datetime_accuracy(timesteps, unit="s"):
    """Check the accuracy of the numpy datetime array.

    Parameters
    ----------
    timesteps : numpy array
        Numpy array to check.
    unit : str, optional
        Unit, by default "s"

    Returns
    -------
    numpy array
        Numpy array checked.

    Raises
    ------
    ValueError
    """
    if not timesteps.dtype == f"<M8[{unit}]":
        msg = f"The timesteps does not have datetime64 {unit} accuracy."
        log_error(logger, msg=msg, verbose=False)
        raise ValueError(msg)
    return timesteps


def _check_timestep_string_second_accuracy(timesteps, n=19):
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
        log_error(logger, msg=msg, verbose=False)
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
        log_error(logger, msg=msg, verbose=False)
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
    if is_numpy_array_string(timesteps):
        timesteps = _check_timesteps_string(timesteps)
    # If numpy datetime64, check accordingly
    elif is_numpy_array_datetime(timesteps):
        timesteps = _check_timestep_datetime_accuracy(timesteps, unit="s")
    else:
        raise TypeError("Unvalid timesteps input.")
    return timesteps


def _check_time_period_nested_list_format(time_periods):
    """Check that the time_periods is a list of list of length 2."""

    if not isinstance(time_periods, list):
        msg = "'time_periods' must be a list'"
        log_error(logger, msg=msg, verbose=False)
        raise TypeError(msg)

    for time_period in time_periods:
        if not isinstance(time_period, (list, np.ndarray)) or len(time_period) != 2:
            msg = "Every time period of time_periods must be a list of length 2."
            log_error(logger, msg=msg, verbose=False)
            raise ValueError(msg)
    return None


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
            msg = f"The {time_period} time_period is unvalid. Start time occurs after end time."
            log_error(logger, msg=msg, verbose=False)
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
    """Check validity of the issue dictionary"""
    # Check is empty
    if len(issue_dict) == 0:
        return issue_dict
    # Check there are only timesteps and time_periods keys
    valid_keys = ["timesteps", "time_periods"]
    keys = list(issue_dict.keys())
    unvalid_keys = [k for k in keys if k not in valid_keys]
    if len(unvalid_keys) > 0:
        msg = f"Unvalid {unvalid_keys} keys. The issue YAML file accept only {valid_keys}"
        log_error(logger, msg=msg, verbose=False)
        raise ValueError(msg)

    # Check timesteps
    timesteps = _get_issue_timesteps(issue_dict)
    # Check time periods
    time_periods = _get_issue_time_periods(issue_dict)
    # Recreate issue dict
    issue_dict["timesteps"] = timesteps
    issue_dict["time_periods"] = time_periods

    return issue_dict


def check_issue_file(fpath: str) -> None:
    """Check issue YAML file validity.

    Parameters
    ----------
    fpath : str
        Issue YAML file path.

    """
    issue_dict = load_yaml_without_date_parsing(fpath)
    issue_dict = check_issue_dict(issue_dict)
    return None


####--------------------------------------------------------------------------.
#### Writer


def _write_issue_docs(f):
    """Provide template for issue.yml"""
    f.write(
        """# This file is used to store timesteps/time periods with wrong/corrupted observation.
# The specified timesteps are dropped during the L0 processing.
# The time format used is the isoformat : YYYY-mm-dd HH:MM:SS.
# The 'timesteps' key enable to specify the list of timesteps to be discarded.
# The 'time_period' key enable to specify the time periods to be dropped.
# Example:
#
# timesteps:
# - 2018-12-07 14:15:00
# - 2018-12-07 14:17:00
# - 2018-12-07 14:19:00
# - 2018-12-07 14:25:00
# time_period:
# - ['2018-08-01 12:00:00', '2018-08-01 14:00:00']
# - ['2018-08-01 15:44:30', '2018-08-01 15:59:31']
# - ['2018-08-02 12:44:30', '2018-08-02 12:59:31'] \n
"""
    )
    return None


def _write_issue(fpath: str, timesteps: list = None, time_periods: list = None) -> None:
    """Write the issue YAML file.

    Parameters
    ----------
    fpath : str
        Filepath of the issue YAML to write.
    timesteps : list, optional
        List of timesteps (to be dropped in L0 processing).
        The default is None.
    time_periods : list, optional
        A list of time periods (to be dropped in L0 processing).
        The default is None.
    """
    # Preprocess timesteps and time_periods (to plain list of strings)
    if timesteps is not None:
        timesteps = timesteps.astype(str).tolist()

    if time_periods is not None:
        new_periods = []
        for i, time_period in enumerate(time_periods):
            new_periods.append(time_period.astype(str).tolist())
        time_periods = new_periods

    # Write the issue YAML file
    logger.info(f"Creating issue YAML file at {fpath}")
    with open(fpath, "w") as f:
        # Write the docs for the issue.yml
        _write_issue_docs(f)

        # Write timesteps if provided
        if timesteps is not None:
            timesteps_dict = {"timesteps": timesteps}
            yaml.dump(timesteps_dict, f, default_flow_style=False)

        # Write time_periods if provided
        if time_periods is not None:
            time_periods_dict = {"time_periods": time_periods}
            yaml.dump(time_periods_dict, f, default_flow_style=None)
    return None


def write_issue_dict(fpath: str, issue_dict: dict) -> None:
    """Write the issue YAML file.

    Parameters
    ----------
    fpath : str
        Filepath of the issue YAML to write.
    issue_dict : dict
        Issue dictionary
    """
    _write_issue(
        fpath=fpath,
        timesteps=issue_dict.get("timesteps", None),
        time_periods=issue_dict.get("time_periods", None),
    )


def write_default_issue(fpath: str) -> None:
    """Write an empty issue YAML file.

    Parameters
    ----------
    fpath : str
        Filepath of the issue YAML to write.
    """
    _write_issue(fpath=fpath)
    return None


####--------------------------------------------------------------------------.
#### Reader


class NoDatesSafeLoader(yaml.SafeLoader):
    @classmethod
    def remove_implicit_resolver(cls, tag_to_remove):
        """
        Remove implicit resolvers for a particular tag

        Takes care not to modify resolvers in super classes.

        We want to load datetimes as strings, not dates, because we
        go on to serialise as json which doesn't have the advanced types
        of yaml, and leads to incompatibilities down the track.
        """
        if "yaml_implicit_resolvers" not in cls.__dict__:
            cls.yaml_implicit_resolvers = cls.yaml_implicit_resolvers.copy()

        for first_letter, mappings in cls.yaml_implicit_resolvers.items():
            cls.yaml_implicit_resolvers[first_letter] = [
                (tag, regexp) for tag, regexp in mappings if tag != tag_to_remove
            ]


def load_yaml_without_date_parsing(filepath):
    "Read a YAML file without converting automatically date string to datetime."
    NoDatesSafeLoader.remove_implicit_resolver("tag:yaml.org,2002:timestamp")
    with open(filepath, "r") as f:
        dictionary = yaml.load(f, Loader=NoDatesSafeLoader)
    # Return empty dictionary if nothing to be read in the file
    if isinstance(dictionary, type(None)):
        dictionary = {}
    return dictionary


def read_issue(raw_dir: str, station_name: str) -> dict:
    """Read YAML issue file.

    Parameters
    ----------
    raw_dir : str
        Path of the campaign raw directory.
    station_name : int
        Station name.

    Returns
    -------
    dict
        Issue dictionary.
    """

    issue_fpath = os.path.join(raw_dir, "issue", station_name + ".yml")
    issue_dict = load_yaml_without_date_parsing(issue_fpath)
    issue_dict = check_issue_dict(issue_dict)
    return issue_dict


def read_issue_file(fpath: str) -> dict:
    """Read YAML issue file.

    Parameters
    ----------
    fpath : str
        Filepath of the issue YAML.

    Returns
    -------
    dict
        Issue dictionary.
    """
    issue_dict = load_yaml_without_date_parsing(fpath)
    issue_dict = check_issue_dict(issue_dict)
    return issue_dict


####--------------------------------------------------------------------------.
