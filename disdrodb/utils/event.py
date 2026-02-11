# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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

import numpy as np
import pandas as pd

from disdrodb.utils.time import ensure_sorted_by_time, temporal_resolution_to_seconds


def split_into_events(
    ds,
    variable,
    *,
    threshold=None,
    neighbor_min_size=2,
    neighbor_time_interval="5MIN",
    event_max_time_gap="6H",
    event_min_duration="5MIN",
    event_min_size=3,
    sortby=None,
    sortby_order="decreasing",
):
    """Split a dataset into "events" and yield each event as a Dataset.

    Events are detected from *candidate timesteps* and then grouped into contiguous events using
    `group_timesteps_into_event`. Candidate timesteps can be selected either by thresholding a numeric
    variable or by using a boolean variable.

    Detection logic
    ---------------
    If ``threshold`` is not None, a timestep is a candidate when ``ds[variable] > threshold``.
    If ``threshold`` is None, ``ds[variable]`` must be boolean;
    a timestep is a candidate when ``ds[variable]`` is True.

    Neighborhood and grouping
    -------------------------
    Candidate timesteps are first filtered for isolation: a candidate is kept only if it has at least
    ``neighbor_min_size`` candidates within ``neighbor_time_interval`` (before/after). Remaining
    candidates are grouped into events when consecutive candidates are separated by no more than
    ``event_max_time_gap``. Events shorter than ``event_min_duration`` or with fewer than ``event_min_size``
    timesteps are discarded.

    Sorting
    -------
    Events are yielded in the grouping order (time order) unless ``sortby`` is provided:
    - ``sortby=None``: yield events in time order (as returned by the grouping).
    - ``sortby="duration"``: sort by event duration.
    - ``sortby`` callable: ``sortby(ds_event) -> scalar`` used as sorting key.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with a ``time`` coordinate/dimension. The dataset is sorted by time internally.
    variable : str
        Name of the variable used for event detection:

        - numeric variable if ``threshold`` is not None
        - boolean variable if ``threshold`` is None


    threshold : int or float or None, optional
        Threshold used to define candidate timesteps.

        - If not None: candidates are where ``ds[variable] > threshold``.
        - If None: ``ds[variable]`` must be boolean and candidates are where it is True.

        Default is None.
    neighbor_time_interval : str
        The time interval around a given a timestep defining the neighborhood.
        Only timesteps that fall within this time interval before or after a timestep are considered neighbors.
        The neighbor_time_interval must be at least equal to the dataset sampling interval!
    neighbor_min_size : int, optional
        The minimum number of neighboring timesteps required within `neighbor_time_interval` for a
        timestep to be considered non-isolated.  Isolated timesteps are removed !

        - If ``neighbor_min_size=0``, then no timestep is considered isolated and no filtering occurs.
        - If ``neighbor_min_size=1``, the timestep must have at least one neighbor within ``neighbor_time_interval``.
        - If ``neighbor_min_size=2``, the timestep must have at least two timesteps within ``neighbor_time_interval``.

        Defaults to 2.
    event_max_time_gap: str
        The maximum time interval between two timesteps to be considered part of the same event.
        This parameters is used to group timesteps into events !
    event_min_duration : str
        The minimum duration an event must span. Events shorter than this duration are discarded.
    event_min_size : int, optional
        The minimum number of valid timesteps required for an event. Defaults to 3.
    sortby: None, str or callable
        Sorting key for events:

        - None: no sorting (time order)
        - "duration": sort by event duration
        - callable: ``sortby(ds_event) -> scalar``

    sortby_order: str
        Sorting direction when ``sortby`` is not None. Default is "decreasing".
        Valid values are "increasing" or "decreasing".

    Yields
    ------
    ds_event : xarray.Dataset
        A view of the input dataset restricted to the event time span

    Notes
    -----
    - This function *yields* event datasets (generator). Use ``list(split_into_events(...))`` to
      materialize all events.
    - Event detection uses ``> threshold`` (strictly greater).

    Examples
    --------
    Threshold-based detection (numeric variable) (timesteps with N > 10)
    >>> events = list(split_into_events(ds, variable="N", threshold=10))

    Boolean-based detection (precomputed mask)
    >>> ds["is_rainy"] = (ds["R"] > 0.1) & (ds["Nbins"] > 2)
    >>> events = list(split_into_events(ds, variable="is_rainy"))

    Sort by duration (longest first)
    >>> for ds_event in split_into_events(ds, variable="N", threshold=10,
                                          sortby="duration", sortby_order="decreasing")
    ...     print(ds_event.time.values[0], ds_event.time.values[-1])

    Sort by a custom scalar (e.g., maximum R during the event)
    >>> sortby_func = lambda ds_event: float(ds_event["R"].max(dim="time").item())
    >>> for ds_event in split_into_events(ds,  variable="N", threshold=10,
                                       sortby=sortby_func, sortby_order="decreasing"):
    ...     print(float(ds_event["R"].max()))
    """
    # Sort dataset by time
    ds = ensure_sorted_by_time(ds)

    # Check neighborhood window >= sampling interval (if available)
    if "sample_interval" in ds:
        sample_interval = ds["sample_interval"].compute().item()
        if temporal_resolution_to_seconds(neighbor_time_interval) < sample_interval:
            msg = "'neighbor_time_interval' must be at least equal to the dataset sample interval ({sample_interval} s)"
            raise ValueError(msg)

    # Define candidate timesteps to group into events
    arr = ds[variable].to_numpy()
    if threshold is None:
        if arr.dtype != np.bool_:
            raise ValueError(
                "When threshold=None, `variable` must be a boolean DataArray. "
                "Provide a boolean variable (True/False) or set threshold to a number.",
            )
        idx_valid = arr
    else:
        idx_valid = arr > threshold

    timesteps = ds["time"].to_numpy()[idx_valid]
    if timesteps.size == 0:
        return iter(())

    # Define event list
    # A list of events, where each event is represented as a dictionary with keys:
    #    - "start_time": np.datetime64, start time of the event
    #    - "end_time": np.datetime64, end time of the event
    #    - "duration": np.timedelta64, duration of the event
    #    - "n_timesteps": int, number of valid timesteps in the event
    event_list = group_timesteps_into_event(
        timesteps=timesteps,
        neighbor_min_size=neighbor_min_size,
        neighbor_time_interval=neighbor_time_interval,
        event_max_time_gap=event_max_time_gap,
        event_min_duration=event_min_duration,
        event_min_size=event_min_size,
    )

    if not event_list:
        return iter(())

    # Slice datasets for each event
    event_ds = []
    for event_info in event_list:
        ds_event = ds.sel(time=slice(event_info["start_time"], event_info["end_time"]))
        if ds_event.sizes.get("time", 0) > 0:
            ds_event["duration"] = event_info["duration"]
            event_ds.append(ds_event)

    if not event_ds:
        return iter(())

    # Optional single-key sort
    if sortby is not None:
        reverse = sortby_order == "decreasing"

        if sortby == "duration":
            event_ds.sort(
                key=lambda ds_event: ds_event["duration"].item() / np.timedelta64(1, "s"),
                reverse=reverse,
            )
        elif callable(sortby):
            # Check that sortby is a callable that accepts a dataset and returns a scalar
            if np.size(sortby(event_ds[0])) != 1:
                raise ValueError("When sortby is a callable, it must return a scalar value.")

            event_ds.sort(key=lambda ds_event: float(sortby(ds_event)), reverse=reverse)
        else:
            raise ValueError("sortby must be None, 'duration', or a callable(ds_event)->scalar.")

    # Yield only the event datasets
    for ds_event in event_ds:
        yield ds_event


def group_timesteps_into_event(
    timesteps,
    event_max_time_gap,
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
    exceed `event_max_time_gap`. Finally, events that do not meet minimum size or duration
    requirements are filtered out.

    Please note that neighbor_min_size and neighbor_time_interval are very sensitive to the
    actual sample interval of the data !

    Parameters
    ----------
    timesteps: numpy.ndarray
        Candidate timesteps to be grouped into events.
    neighbor_time_interval : str
        The time interval around a given a timestep defining the neighborhood.
        Only timesteps that fall within this time interval before or after a timestep are considered neighbors.
    neighbor_min_size : int, optional
        The minimum number of neighboring timesteps required within `neighbor_time_interval` for a
        timestep to be considered non-isolated.  Isolated timesteps are removed !

        - If ``neighbor_min_size=0``, then no timestep is considered isolated and no filtering occurs.
        - If ``neighbor_min_size=1``, the timestep must have at least one neighbor within ``neighbor_time_interval``.
        - If ``neighbor_min_size=2``, the timestep must have at least two timesteps within ``neighbor_time_interval``.

        Defaults to 1.
    event_max_time_gap: str
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
    neighbor_time_interval = pd.Timedelta(temporal_resolution_to_seconds(neighbor_time_interval), unit="seconds")
    event_max_time_gap = pd.Timedelta(temporal_resolution_to_seconds(event_max_time_gap), unit="seconds")
    event_min_duration = pd.Timedelta(temporal_resolution_to_seconds(event_min_duration), unit="seconds")

    # Remove isolated timesteps
    timesteps = remove_isolated_timesteps(
        timesteps,
        neighbor_min_size=neighbor_min_size,
        neighbor_time_interval=neighbor_time_interval,
    )

    # Group timesteps into events
    # - If two timesteps are separated by less than event_max_time_gap, are considered the same event
    events = group_timesteps_into_events(timesteps, event_max_time_gap)

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
    timesteps : array-like of numpy.datetime64
        Sorted or unsorted array of valid timesteps.
    neighbor_time_interval : numpy.timedelta64
        The time interval around a given a timestep defining the neighborhood.
        Only timesteps that fall within this time interval before or after a timestep are considered neighbors.
    neighbor_min_size : int, optional
        The minimum number of neighboring timesteps required within `neighbor_time_interval` for a
        timestep to be considered non-isolated.

        - If ``neighbor_min_size=0``, then no timestep is considered isolated and no filtering occurs.
        - If ``neighbor_min_size=1``, the timestep must have at least one neighbor within ``neighbor_time_interval``.
        - If ``neighbor_min_size=2``, the timestep must have at least two timesteps within ``neighbor_time_interval``.

        Defaults to 1.

    Returns
    -------
    numpy.ndarray
        Array of timesteps with isolated entries removed.
    """
    # Sort timesteps
    timesteps = np.array(timesteps)
    timesteps = np.sort(timesteps)

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


def group_timesteps_into_events(timesteps, event_max_time_gap):
    """
    Group valid timesteps into events based on a maximum allowed dry interval.

    Parameters
    ----------
    timesteps : array-like of numpy.datetime64
        Sorted array of valid timesteps.
    event_max_time_gap : numpy.timedelta64
        Maximum time interval allowed between consecutive valid timesteps for them
        to be considered part of the same event.

    Returns
    -------
    list of numpy.ndarray
        A list of events, where each event is an array of timesteps.
    """
    # Deal with case with no timesteps
    if len(timesteps) == 0:
        return []

    # Ensure timesteps are sorted
    timesteps = np.sort(timesteps)

    # Compute differences between consecutive timesteps
    diffs = np.diff(timesteps)

    # Identify the indices where the gap is larger than event_max_time_gap
    # These indices represent boundaries between events
    break_indices = np.where(diffs > event_max_time_gap)[0] + 1

    # Split the timesteps at the identified break points
    events = np.split(timesteps, break_indices)

    # NON VECTORIZED CODE
    # events = []
    # current_event = [timesteps[0]]
    # for i in range(1, len(timesteps)):
    #     current_t = timesteps[i]
    #     previous_t = timesteps[i - 1]

    #     if current_t - previous_t <= event_max_time_gap:
    #         current_event.append(current_t)
    #     else:
    #         events.append(current_event)
    #         current_event = [current_t]

    # events.append(current_event)
    return events


####-----------------------------------------------------------------------------------.
