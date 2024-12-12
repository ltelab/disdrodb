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
"""Functions for event definition."""

import datetime

import dask
import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.api.info import get_start_end_time_from_filepaths
from disdrodb.utils.time import ensure_sorted_by_time


@dask.delayed
def _delayed_open_dataset(filepath):
    with dask.config.set(scheduler="synchronous"):
        ds = xr.open_dataset(filepath, chunks={}, autoclose=True, cache=False)
    return ds


def identify_events(filepaths, parallel):
    """Identify rainy events."""
    # Open datasets in parallel
    if parallel:
        list_ds = dask.compute([_delayed_open_dataset(filepath) for filepath in filepaths])[0]
    else:
        list_ds = [xr.open_dataset(filepath, chunks={}, cache=False) for filepath in filepaths]

    # List sample interval
    sample_intervals = np.array([ds["sample_interval"].data.item() for ds in list_ds])
    # Concat datasets
    ds = xr.concat(list_ds, dim="time")
    # Read in memory what is needed
    ds = ds[["time", "n_drops_selected"]].compute()
    # Close file on disk
    _ = [ds.close() for ds in list_ds]
    del list_ds
    # Check for sample intervals
    if len(set(sample_intervals)) > 1:
        raise ValueError("Sample intervals are not constant across files.")
    # Sort dataset by time
    ds = ensure_sorted_by_time(ds)
    # Select events
    # TODO:
    minimum_n_drops = 5
    min_time_contiguity = pd.Timedelta(datetime.timedelta(minutes=10))
    max_dry_time_interval = pd.Timedelta(datetime.timedelta(hours=2))
    minimum_duration = pd.Timedelta(datetime.timedelta(minutes=5))
    event_list = select_events(
        ds=ds,
        minimum_n_drops=minimum_n_drops,
        minimum_duration=minimum_duration,
        min_time_contiguity=min_time_contiguity,
        max_dry_time_interval=max_dry_time_interval,
    )
    return event_list


def remove_isolated_timesteps(timesteps, min_time_contiguity):
    """Remove isolated timesteps."""
    # Sort timesteps just in case
    timesteps.sort()
    cleaned_timesteps = []
    for i, t in enumerate(timesteps):
        prev_t = timesteps[i - 1] if i > 0 else None
        next_t = timesteps[i + 1] if i < len(timesteps) - 1 else None

        is_isolated = True
        if prev_t and t - prev_t <= min_time_contiguity:
            is_isolated = False
        if next_t and next_t - t <= min_time_contiguity:
            is_isolated = False

        if not is_isolated:
            cleaned_timesteps.append(t)

    return cleaned_timesteps


def group_timesteps_into_events(timesteps, max_dry_time_interval):
    """Group timesteps into events."""
    timesteps.sort()
    events = []
    current_event = [timesteps[0]]

    for i in range(1, len(timesteps)):
        current_t = timesteps[i]
        previous_t = timesteps[i - 1]

        if current_t - previous_t <= max_dry_time_interval:
            current_event.append(current_t)
        else:
            events.append(current_event)
            current_event = [current_t]

    events.append(current_event)
    return events


def select_events(
    ds,
    minimum_n_drops,
    minimum_duration,  # TODO UNUSED
    min_time_contiguity,
    max_dry_time_interval,
):
    """Select events."""
    timesteps = ds["time"].data
    n_drops_selected = ds["n_drops_selected"].data

    # Define candidate timesteps to group into events
    idx_valid = n_drops_selected > minimum_n_drops
    timesteps = timesteps[idx_valid]

    # Remove noisy timesteps
    timesteps = remove_isolated_timesteps(timesteps, min_time_contiguity)

    # Group timesteps into events
    events = group_timesteps_into_events(timesteps, max_dry_time_interval)

    # Define list of event
    event_list = [
        {
            "start_time": event[0],
            "end_time": event[-1],
            "duration": (event[-1] - event[0]).astype("m8[m]"),
        }
        for event in events
    ]
    return event_list


def get_events_info(list_events, filepaths, accumulation_interval, rolling):
    """
    Provide information about the required files for each event.

    For each event in `list_events`, this function identifies the file paths from `filepaths` that
    overlap with the event period, adjusted by the `accumulation_interval`. The event period is
    extended backward or forward based on the `rolling` parameter.

    Parameters
    ----------
    list_events : list of dict
        List of events, where each event is a dictionary containing at least 'start_time' and 'end_time'
        keys with `numpy.datetime64` values.
    filepaths : list of str
        List of file paths corresponding to data files.
    accumulation_interval : numpy.timedelta64 or int
        Time interval to adjust the event period for accumulation. If an integer is provided, it is
        assumed to be in seconds.
    rolling : bool
        If True, adjust the event period backward by `accumulation_interval` (rolling backward).
        If False, adjust forward (aggregate forward).

    Returns
    -------
    list of dict
        A list where each element is a dictionary containing:
        - 'start_time': Adjusted start time of the event (`numpy.datetime64`).
        - 'end_time': Adjusted end time of the event (`numpy.datetime64`).
        - 'filepaths': List of file paths overlapping with the adjusted event period.

    """
    # Ensure accumulation_interval is numpy.timedelta64
    if not isinstance(accumulation_interval, np.timedelta64):
        accumulation_interval = np.timedelta64(accumulation_interval, "s")

    # Retrieve file start_time and end_time
    files_start_time, files_end_time = get_start_end_time_from_filepaths(filepaths)

    # Retrieve information for each event
    event_info = []
    for event_dict in list_events:
        # Retrieve event time period
        event_start_time = event_dict["start_time"]
        event_end_time = event_dict["end_time"]

        # Add buffer to account for accumulation interval
        if rolling:  # backward
            event_start_time = event_start_time - np.array(accumulation_interval, dtype="m8[s]")
        else:  # aggregate forward
            event_end_time = event_end_time + np.array(accumulation_interval, dtype="m8[s]")

        # Derive event filepaths
        overlaps = (files_start_time <= event_end_time) & (files_end_time >= event_start_time)
        event_filepaths = np.array(filepaths)[overlaps].tolist()

        # Create dictionary
        if len(event_filepaths) > 0:
            event_info.append(
                {"start_time": event_start_time, "end_time": event_end_time, "filepaths": event_filepaths},
            )

    return event_info


# list_events[0]
# accumulation_interval = 30


# For event
# - Get start_time, end_time
# - Buffer start_time and end_time by accumulation_interval

# - Get filepaths for start_time, end_time (assign_filepaths_to_event)
