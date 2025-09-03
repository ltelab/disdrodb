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
"""Utility function for DISDRODB product archiving."""
import numpy as np
import pandas as pd

from disdrodb.api.info import get_start_end_time_from_filepaths
from disdrodb.api.io import open_netcdf_files
from disdrodb.utils.event import group_timesteps_into_event
from disdrodb.utils.time import (
    ensure_sorted_by_time,
    ensure_timedelta_seconds,
)

####---------------------------------------------------------------------------------
#### Time blocks


def check_freq(freq: str) -> None:
    """Check validity of freq argument."""
    valid_freq = ["none", "year", "season", "quarter", "month", "day", "hour"]
    if not isinstance(freq, str):
        raise TypeError("'freq' must be a string.")
    if freq not in valid_freq:
        raise ValueError(
            f"'freq' '{freq}' is not possible. Must be one of: {valid_freq}.",
        )
    return freq


def generate_time_blocks(start_time: np.datetime64, end_time: np.datetime64, freq: str) -> np.ndarray:  # noqa: PLR0911
    """Generate time blocks between `start_time` and `end_time` for a given frequency.

    Parameters
    ----------
    start_time : numpy.datetime64
        Inclusive start of the overall time range.
    end_time : numpy.datetime64
        Inclusive end of the overall time range.
    freq : str
        Frequency specifier. Accepted values are:
        - 'none'    : return a single block [start_time, end_time]
        - 'day'     : split into daily blocks
        - 'month'   : split into calendar months
        - 'quarter' : split into calendar quarters
        - 'year'    : split into calendar years
        - 'season'  : split into meteorological seasons (MAM, JJA, SON, DJF)

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 2) with dtype datetime64[s], where each row is [block_start, block_end].

    """
    freq = check_freq(freq)
    if freq == "none":
        return np.array([[start_time, end_time]], dtype="datetime64[s]")

    if freq == "hour":
        periods = pd.period_range(start=start_time, end=end_time, freq="h")
        blocks = np.array(
            [
                [
                    period.start_time.to_datetime64().astype("datetime64[s]"),
                    period.end_time.to_datetime64().astype("datetime64[s]"),
                ]
                for period in periods
            ],
            dtype="datetime64[s]",
        )
        return blocks

    if freq == "day":
        periods = pd.period_range(start=start_time, end=end_time, freq="d")
        blocks = np.array(
            [
                [
                    period.start_time.to_datetime64().astype("datetime64[s]"),
                    period.end_time.to_datetime64().astype("datetime64[s]"),
                ]
                for period in periods
            ],
            dtype="datetime64[s]",
        )
        return blocks

    if freq == "month":
        periods = pd.period_range(start=start_time, end=end_time, freq="M")
        blocks = np.array(
            [
                [
                    period.start_time.to_datetime64().astype("datetime64[s]"),
                    period.end_time.to_datetime64().astype("datetime64[s]"),
                ]
                for period in periods
            ],
            dtype="datetime64[s]",
        )
        return blocks

    if freq == "year":
        periods = pd.period_range(start=start_time, end=end_time, freq="Y")
        blocks = np.array(
            [
                [
                    period.start_time.to_datetime64().astype("datetime64[s]"),
                    period.end_time.to_datetime64().astype("datetime64[s]"),
                ]
                for period in periods
            ],
            dtype="datetime64[s]",
        )
        return blocks

    if freq == "quarter":
        periods = pd.period_range(start=start_time, end=end_time, freq="Q")
        blocks = np.array(
            [
                [
                    period.start_time.to_datetime64().astype("datetime64[s]"),
                    period.end_time.floor("s").to_datetime64().astype("datetime64[s]"),
                ]
                for period in periods
            ],
            dtype="datetime64[s]",
        )
        return blocks

    # if freq == "season":
    # Fiscal quarter frequency ending in Feb → seasons DJF, MAM, JJA, SON
    periods = pd.period_range(start=start_time, end=end_time, freq="Q-FEB")
    blocks = np.array(
        [
            [
                period.start_time.to_datetime64().astype("datetime64[s]"),
                period.end_time.to_datetime64().astype("datetime64[s]"),
            ]
            for period in periods
        ],
        dtype="datetime64[s]",
    )
    return blocks


####----------------------------------------------------------------------------
#### Event/Time partitioning
def identify_events(
    filepaths,
    parallel=False,
    min_drops=5,
    neighbor_min_size=2,
    neighbor_time_interval="5MIN",
    event_max_time_gap="6H",
    event_min_duration="5MIN",
    event_min_size=3,
):
    """Return a list of rainy events.

    Rainy timesteps are defined when N > min_drops.
    Any rainy isolated timesteps (based on neighborhood criteria) is removed.
    Then, consecutive rainy timesteps are grouped into the same event if the time gap between them does not
    exceed `event_max_time_gap`. Finally, events that do not meet minimum size or duration
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
    # Open datasets in parallel
    ds = open_netcdf_files(filepaths, variables=["time", "N"], parallel=parallel, compute=True)
    # Sort dataset by time
    ds = ensure_sorted_by_time(ds)
    # Define candidate timesteps to group into events
    idx_valid = ds["N"].to_numpy() > min_drops
    timesteps = ds["time"].to_numpy()[idx_valid]
    # Define event list
    event_list = group_timesteps_into_event(
        timesteps=timesteps,
        neighbor_min_size=neighbor_min_size,
        neighbor_time_interval=neighbor_time_interval,
        event_max_time_gap=event_max_time_gap,
        event_min_duration=event_min_duration,
        event_min_size=event_min_size,
    )
    del ds
    return event_list


def identify_time_partitions(filepaths: list[str], freq: str) -> list[dict]:
    """Identify the set of time blocks covered by files.

    The result is a minimal, sorted, and unique set of time partitions.

    Parameters
    ----------
    filepaths : list of str
        Paths to input files from which start and end times will be extracted
        via `get_start_end_time_from_filepaths`.
    freq : {'none', 'hour', 'day', 'month', 'quarter', 'season', 'year'}
        Frequency determining the granularity of candidate blocks.
        See `generate_time_blocks` for more details.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing:

        - `start_time` (numpy.datetime64[s])
            Inclusive start of a time block.
        - `end_time` (numpy.datetime64[s])
            Inclusive end of a time block.

        Only those blocks that overlap at least one file's interval are returned.
        The list is sorted by `start_time` and contains no duplicate blocks.
    """
    # Define file start time and end time
    start_times, end_times = get_start_end_time_from_filepaths(filepaths)

    # Define files time coverage
    start_time, end_time = start_times.min(), end_times.max()

    # Compute candidate time blocks
    blocks = generate_time_blocks(start_time, end_time, freq=freq)  # TODO end_time non inclusive is correct?

    # Select time blocks with files
    mask = (blocks[:, 0][:, None] <= end_times) & (blocks[:, 1][:, None] >= start_times)
    blocks = blocks[mask.any(axis=1)]

    # Ensure sorted unique time blocks
    order = np.argsort(blocks[:, 0])
    blocks = np.unique(blocks[order], axis=0)

    # Convert to list of dicts
    list_time_blocks = [{"start_time": start_time, "end_time": end_time} for start_time, end_time in blocks]
    return list_time_blocks


def define_temporal_partitions(filepaths, strategy, parallel, strategy_options):
    """Define temporal file processing partitions.

    Parameters
    ----------
    filepaths : list
        List of files paths to be processed

    strategy : str
        Which partitioning strategy to apply:

        - ``'time_block'`` defines fixed time intervals (e.g. monthly) covering input files.
        - ``'event'`` detect clusters of precipitation ("events").

    parallel : bool
         If True, parallel data loading is used to identify events.

    strategy_options : dict
        Dictionary with strategy-specific parameters:

        If ``strategy == 'time_block'``, supported options are:

        - ``freq``: Time unit for blocks. One of {'year', 'season', 'month', 'day'}.

        See identify_time_partitions for more information.

        If ``strategy == 'event'``, supported options are:

        - ``min_drops`` : int
          Minimum number of drops to consider a timestep.
        - ``neighbor_min_size`` : int
          Minimum cluster size for merging neighboring events.
        - ``neighbor_time_interval`` : str
          Time window (e.g. "5MIN") to merge adjacent clusters.
        - ``event_max_time_gap`` : str
          Maximum allowed gap (e.g. "6H") within a single event.
        - ``event_min_duration`` : str
          Minimum total duration (e.g. "5MIN") of an event.
        - ``event_min_size`` : int
          Minimum number of records in an event.

        See identify_events for more information.

    Returns
    -------
    list
        A list of dictionaries, each containing:

        - ``start_time`` (numpy.datetime64[s])
            Inclusive start of an event or time block.
        - ``end_time`` (numpy.datetime64[s])
            Inclusive end of an event or time block.

    Notes
    -----
    - The ``'event'`` strategy requires loading data into memory to identify clusters.
    - The ``'time_block'`` strategy can operate on metadata alone, without full data loading.
    - The ``'event'`` strategy implicitly performs data selection on which files to process !
    - The ``'time_block'`` strategy does not performs data selection on which files to process !
    """
    if strategy not in ["time_block", "event"]:
        raise ValueError(f"Unknown strategy: {strategy!r}. Must be 'time_block' or 'event'.")
    if strategy == "event":
        return identify_events(filepaths, parallel=parallel, **strategy_options)

    return identify_time_partitions(filepaths, **strategy_options)


####----------------------------------------------------------------------------
#### Filepaths partitioning


def get_files_partitions(list_partitions, filepaths, sample_interval, accumulation_interval, rolling):  # noqa: ARG001
    """
    Provide information about the required files for each event.

    For each event in `list_partitions`, this function identifies the file paths from `filepaths` that
    overlap with the event period, adjusted by the `accumulation_interval`. The event period is
    extended backward or forward based on the `rolling` parameter.

    Parameters
    ----------
    list_partitions : list of dict
        List of events, where each event is a dictionary containing at least 'start_time' and 'end_time'
        keys with `numpy.datetime64` values.
    filepaths : list of str
        List of file paths corresponding to data files.
    sample_interval : numpy.timedelta64 or int
        The sample interval of the input dataset.
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
    # Ensure sample_interval and accumulation_interval is numpy.timedelta64
    accumulation_interval = ensure_timedelta_seconds(accumulation_interval)
    sample_interval = ensure_timedelta_seconds(sample_interval)

    # Retrieve file start_time and end_time
    files_start_time, files_end_time = get_start_end_time_from_filepaths(filepaths)

    # Retrieve information for each event
    event_info = []
    for event_dict in list_partitions:
        # Retrieve event time period
        event_start_time = event_dict["start_time"]
        event_end_time = event_dict["end_time"]

        # Adapt event_end_time if accumulation interval different from sample interval
        if sample_interval != accumulation_interval:
            event_end_time = event_end_time + accumulation_interval

        # Derive event filepaths
        overlaps = (files_start_time <= event_end_time) & (files_end_time >= event_start_time)
        event_filepaths = np.array(filepaths)[overlaps].tolist()

        # Create dictionary
        if len(event_filepaths) > 0:
            event_info.append(
                {"start_time": event_start_time, "end_time": event_end_time, "filepaths": event_filepaths},
            )

    return event_info


def get_files_per_time_block(filepaths, freq="day", tolerance_seconds=0):
    """
    Organize files by the time blocks they cover (day, month, etc.).

    Parameters
    ----------
    filepaths : list of str
        List of file paths to be processed.
    freq : {'none', 'hour', 'day', 'month', 'quarter', 'season', 'year'}
        Frequency used to define the time blocks.

    Returns
    -------
    dict
        Dictionary where keys are the time block identifiers (as strings)
        and values are lists of file paths that overlap with that block.
    """
    if not filepaths:
        return {}

    # Identify candidate blocks
    list_partitions = identify_time_partitions(filepaths, freq=freq)

    # Retrieve file start_time and end_time
    files_start_time, files_end_time = get_start_end_time_from_filepaths(filepaths)

    # Add tolerance for sensor imprecision
    files_start_time = files_start_time - np.array(tolerance_seconds, dtype="m8[s]")
    files_end_time = files_end_time + np.array(tolerance_seconds, dtype="m8[s]")

    # Retrieve time blocks start & end
    block_starts = np.array([b["start_time"] for b in list_partitions])
    block_ends = np.array([b["end_time"] for b in list_partitions])

    # Broadcast to (n_files, n_blocks)
    mask = (files_start_time[:, None] <= block_ends[None, :]) & (files_end_time[:, None] >= block_starts[None, :])

    # Build dictionary
    dict_blocks = {}
    filepaths = np.array(filepaths)
    for j, (start, end) in enumerate(zip(block_starts, block_ends)):
        file_indices = np.where(mask[:, j])[0]
        if file_indices.size > 0:
            if freq == "none":
                block_key = str(start)
            else:
                # Pick a stable representation (use block start as key)
                block_key = str(start)
            dict_blocks[block_key] = filepaths[file_indices].tolist()

    return dict_blocks


def get_files_per_days(filepaths):
    """
    Organize files by the days they cover based on their start and end times.

    Parameters
    ----------
    filepaths : list of str
        List of file paths to be processed.

    Returns
    -------
    dict
        Dictionary where keys are days (as strings) and values are lists of file paths
        that cover those days.

    Notes
    -----
    This function adds a tolerance of 60 seconds to account for imprecise time logging by the sensors.
    """
    # Empty filepaths list return a dictionary
    if len(filepaths) == 0:
        return {}

    # Retrieve file start_time and end_time
    files_start_time, files_end_time = get_start_end_time_from_filepaths(filepaths)

    # Add tolerance to account for imprecise time logging by the sensors
    # - Example: timestep 23:59:30 might be 00.00 and goes into the next day file ...
    files_start_time = files_start_time - np.array(TOLERANCE_SECONDS, dtype="m8[s]")
    files_end_time = files_end_time + np.array(TOLERANCE_SECONDS, dtype="m8[s]")

    # Retrieve file start day and end day
    start_day = files_start_time.min().astype("M8[D]")
    end_day = files_end_time.max().astype("M8[D]") + np.array(1, dtype="m8[D]")

    # Create an array with all days in time period covered by the files
    list_days = np.asanyarray(pd.date_range(start=start_day, end=end_day, freq="D")).astype("M8[D]")

    # Expand dimension to match each day using broadcasting
    files_start_time = files_start_time.astype("M8[D]")[:, np.newaxis]  # shape (n_files, 1)
    files_end_time = files_end_time.astype("M8[D]")[:, np.newaxis]  # shape (n_files, 1)

    # Create an array of all days
    # - Expand dimension to match each day using broadcasting
    days = list_days[np.newaxis, :]  # shape (1, n_days)

    # Use broadcasting to create a boolean matrix indicating which files cover which days
    mask = (files_start_time <= days) & (files_end_time >= days)  # shape (n_files, n_days)

    # Build a mapping from days to file indices
    # For each day (column), find the indices of files (rows) that cover that day
    dict_days = {}
    filepaths = np.array(filepaths)
    for i, day in enumerate(list_days):
        file_indices = np.where(mask[:, i])[0]
        if file_indices.size > 0:
            dict_days[str(day)] = filepaths[file_indices].tolist()

    return dict_days
