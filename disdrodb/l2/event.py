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
import dask
import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.api.info import get_start_end_time_from_filepaths
from disdrodb.utils.time import acronym_to_seconds, ensure_sorted_by_time


@dask.delayed
def _delayed_open_dataset(filepath):
    with dask.config.set(scheduler="synchronous"):
        ds = xr.open_dataset(filepath, chunks={}, autoclose=True, decode_timedelta=False, cache=False)
    return ds


def identify_events(
    filepaths,
    parallel=False,
    min_n_drops=5,
    neighbor_min_size=2,
    neighbor_time_interval="5MIN",
    intra_event_max_time_gap="6H",
    event_min_duration="5MIN",
    event_min_size=3,
):
    """Return a list of rainy events.

    Rainy timesteps are defined when N > min_n_drops.
    Any rainy isolated timesteps (based on neighborhood criteria) is removed.
    Then, consecutive rainy timesteps are grouped into the same event if the time gap between them does not
    exceed `intra_event_max_time_gap`. Finally, events that do not meet minimum size or duration
    requirements are filtered out.

    Parameters
    ----------
    filepaths: list
        List of L1C file paths.
    parallel: bool
        Whether to load the files in parallel.
        Set parallel=True only in a multiprocessing environment.
        The default is False.
    neighbor_time_interval : str
        The time interval around a given a timestep defining the neighborhood.
        Only timesteps that fall within this time interval before or after a timestep are considered neighbors.
    neighbor_min_size : int, optional
        The minimum number of neighboring timesteps required within `neighbor_time_interval` for a
        timestep to be considered non-isolated.  Isolated timesteps are removed !
        - If `neighbor_min_size=0,  then no timestep is considered isolated and no filtering occurs.
        - If `neighbor_min_size=1`, the timestep must have at least one neighbor within `neighbor_time_interval`.
        - If `neighbor_min_size=2`, the timestep must have at least two timesteps within `neighbor_time_interval`.
        Defaults to 1.
    intra_event_max_time_gap: str
        The maximum time interval between two timesteps to be considered part of the same event.
        This parameters is used to group timesteps into events !
    event_min_duration : str
        The minimum duration an event must span. Events shorter than this duration are discarded.
    event_min_size : int, optional
        The minimum number of valid timesteps required for an event. Defaults to 1.

    Returns
    -------
    list of dict
        A list of events, where each event is represented as a dictionary with keys:
        - "start_time": np.datetime64, start time of the event
        - "end_time": np.datetime64, end time of the event
        - "duration": np.timedelta64, duration of the event
        - "n_timesteps": int, number of valid timesteps in the event
    """
    # Open datasets in parallel
    if parallel:
        list_ds = dask.compute([_delayed_open_dataset(filepath) for filepath in filepaths])[0]
    else:
        list_ds = [xr.open_dataset(filepath, chunks={}, cache=False, decode_timedelta=False) for filepath in filepaths]
    # Filter dataset for requested variables
    variables = ["time", "N"]
    list_ds = [ds[variables] for ds in list_ds]
    # Concat datasets
    ds = xr.concat(list_ds, dim="time", compat="no_conflicts", combine_attrs="override")
    # Read in memory the variable needed
    ds = ds.compute()
    # Close file on disk
    _ = [ds.close() for ds in list_ds]
    del list_ds
    # Sort dataset by time
    ds = ensure_sorted_by_time(ds)
    # Define candidate timesteps to group into events
    idx_valid = ds["N"].data > min_n_drops
    timesteps = ds["time"].data[idx_valid]
    # Define event list
    event_list = group_timesteps_into_event(
        timesteps=timesteps,
        neighbor_min_size=neighbor_min_size,
        neighbor_time_interval=neighbor_time_interval,
        intra_event_max_time_gap=intra_event_max_time_gap,
        event_min_duration=event_min_duration,
        event_min_size=event_min_size,
    )
    return event_list


def group_timesteps_into_event(
    timesteps,
    intra_event_max_time_gap,
    event_min_size=0,
    event_min_duration="0S",
    neighbor_min_size=0,
    neighbor_time_interval="0S",
):
    """
    Group candidate timesteps into events based on temporal criteria.

    This function groups valid candidate timesteps into events by considering how they cluster
    in time. Any isolated timesteps (based on neighborhood criteria) are first removed. Then,
    consecutive timesteps are grouped into the same event if the time gap between them does not
    exceed `intra_event_max_time_gap`. Finally, events that do not meet minimum size or duration
    requirements are filtered out.

    Please note that neighbor_min_size and neighbor_time_interval are very sensitive to the
    actual sample interval of the data !

    Parameters
    ----------
    timesteps: np.ndarray
        Candidate timesteps to be grouped into events.
    neighbor_time_interval : str
        The time interval around a given a timestep defining the neighborhood.
        Only timesteps that fall within this time interval before or after a timestep are considered neighbors.
    neighbor_min_size : int, optional
        The minimum number of neighboring timesteps required within `neighbor_time_interval` for a
        timestep to be considered non-isolated.  Isolated timesteps are removed !
        - If `neighbor_min_size=0,  then no timestep is considered isolated and no filtering occurs.
        - If `neighbor_min_size=1`, the timestep must have at least one neighbor within `neighbor_time_interval`.
        - If `neighbor_min_size=2`, the timestep must have at least two timesteps within `neighbor_time_interval`.
        Defaults to 1.
    intra_event_max_time_gap: str
        The maximum time interval between two timesteps to be considered part of the same event.
        This parameters is used to group timesteps into events !
    event_min_duration : str
        The minimum duration an event must span. Events shorter than this duration are discarded.
    event_min_size : int, optional
        The minimum number of valid timesteps required for an event. Defaults to 1.

    Returns
    -------
    list of dict
        A list of events, where each event is represented as a dictionary with keys:
        - "start_time": np.datetime64, start time of the event
        - "end_time": np.datetime64, end time of the event
        - "duration": np.timedelta64, duration of the event
        - "n_timesteps": int, number of valid timesteps in the event
    """
    # Retrieve datetime arguments
    neighbor_time_interval = pd.Timedelta(acronym_to_seconds(neighbor_time_interval), unit="seconds")
    intra_event_max_time_gap = pd.Timedelta(acronym_to_seconds(intra_event_max_time_gap), unit="seconds")
    event_min_duration = pd.Timedelta(acronym_to_seconds(event_min_duration), unit="seconds")

    # Remove isolated timesteps
    timesteps = remove_isolated_timesteps(
        timesteps,
        neighbor_min_size=neighbor_min_size,
        neighbor_time_interval=neighbor_time_interval,
    )

    # Group timesteps into events
    # - If two timesteps are separated by less than intra_event_max_time_gap, are considered the same event
    events = group_timesteps_into_events(timesteps, intra_event_max_time_gap)

    # Define list of event
    event_list = [
        {
            "start_time": event[0],
            "end_time": event[-1],
            "duration": (event[-1] - event[0]).astype("m8[m]"),
            "n_timesteps": len(event),
        }
        for event in events
    ]

    # Filter event list by duration
    event_list = [event for event in event_list if event["duration"] >= event_min_duration]

    # Filter event list by duration
    event_list = [event for event in event_list if event["n_timesteps"] >= event_min_size]

    return event_list


def remove_isolated_timesteps(timesteps, neighbor_min_size, neighbor_time_interval):
    """
    Remove isolated timesteps that do not have enough neighboring timesteps within a specified time gap.

    A timestep is considered isolated (and thus removed) if it does not have at least `neighbor_min_size` other
    timesteps within the `neighbor_time_interval` before or after it.
    In other words, for each timestep, we look for how many other timesteps fall into the
    time interval [t - neighbor_time_interval, t + neighbor_time_interval], excluding it itself.
    If the count of such neighbors is less than `neighbor_min_size`, that timestep is removed.

    Parameters
    ----------
    timesteps : array-like of np.datetime64
        Sorted or unsorted array of valid timesteps.
    neighbor_time_interval : np.timedelta64
        The time interval around a given a timestep defining the neighborhood.
        Only timesteps that fall within this time interval before or after a timestep are considered neighbors.
    neighbor_min_size : int, optional
        The minimum number of neighboring timesteps required within `neighbor_time_interval` for a
        timestep to be considered non-isolated.
        - If `neighbor_min_size=0,  then no timestep is considered isolated and no filtering occurs.
        - If `neighbor_min_size=1`, the timestep must have at least one neighbor within `neighbor_time_interval`.
        - If `neighbor_min_size=2`, the timestep must have at least two timesteps within `neighbor_time_interval`.
        Defaults to 1.

    Returns
    -------
    np.ndarray
        Array of timesteps with isolated entries removed.
    """
    # Sort timesteps
    timesteps = np.array(timesteps)
    timesteps.sort()

    # Do nothing if neighbor_min_size is 0
    if neighbor_min_size == 0:
        return timesteps

    # Compute the start and end of the interval for each timestep
    t_starts = timesteps - neighbor_time_interval
    t_ends = timesteps + neighbor_time_interval

    # Use searchsorted to find the positions where these intervals would be inserted
    # to keep the array sorted. This effectively gives us the bounds of timesteps
    # within the neighbor interval.
    left_indices = np.searchsorted(timesteps, t_starts, side="left")
    right_indices = np.searchsorted(timesteps, t_ends, side="right")

    # The number of neighbors is the difference in indices minus one (to exclude the timestep itself)
    n_neighbors = right_indices - left_indices - 1
    valid_mask = n_neighbors >= neighbor_min_size

    non_isolated_timesteps = timesteps[valid_mask]

    # NON VECTORIZED CODE
    # non_isolated_timesteps = []
    # n_neighbours_arr = []
    # for i, t in enumerate(timesteps):
    #     n_neighbours = np.sum(np.logical_and(timesteps >= (t - neighbor_time_interval),
    #                                          timesteps <= (t + neighbor_time_interval))) - 1
    #     n_neighbours_arr.append(n_neighbours)
    #     if n_neighbours > neighbor_min_size:
    #       non_isolated_timesteps.append(t)
    # non_isolated_timesteps = np.array(non_isolated_timesteps)
    return non_isolated_timesteps


def group_timesteps_into_events(timesteps, intra_event_max_time_gap):
    """
    Group valid timesteps into events based on a maximum allowed dry interval.

    Parameters
    ----------
    timesteps : array-like of np.datetime64
        Sorted array of valid timesteps.
    intra_event_max_time_gap : np.timedelta64
        Maximum time interval allowed between consecutive valid timesteps for them
        to be considered part of the same event.

    Returns
    -------
    list of np.ndarray
        A list of events, where each event is an array of timesteps.
    """
    # Deal with case with no timesteps
    if len(timesteps) == 0:
        return []

    # Ensure timesteps are sorted
    timesteps.sort()

    # Compute differences between consecutive timesteps
    diffs = np.diff(timesteps)

    # Identify the indices where the gap is larger than intra_event_max_time_gap
    # These indices represent boundaries between events
    break_indices = np.where(diffs > intra_event_max_time_gap)[0] + 1

    # Split the timesteps at the identified break points
    events = np.split(timesteps, break_indices)

    # NON VECTORIZED CODE
    # events = []
    # current_event = [timesteps[0]]
    # for i in range(1, len(timesteps)):
    #     current_t = timesteps[i]
    #     previous_t = timesteps[i - 1]

    #     if current_t - previous_t <= intra_event_max_time_gap:
    #         current_event.append(current_t)
    #     else:
    #         events.append(current_event)
    #         current_event = [current_t]

    # events.append(current_event)
    return events


####-----------------------------------------------------------------------------------.


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
