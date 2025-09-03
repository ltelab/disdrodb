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


def generate_time_blocks(
    start_time: np.datetime64,
    end_time: np.datetime64,
    freq: str,
    inclusive_end_time: bool = True,
) -> np.ndarray:
    """Generate time blocks between `start_time` and `end_time` for a given frequency.

    Parameters
    ----------
    start_time : numpy.datetime64
        Inclusive start of the overall time range.
    end_time : numpy.datetime64
        End of the overall time range. Inclusive by default (see inclusive_end_time argument).
    freq : str
        Frequency specifier. Accepted values are:
        - 'none'    : return a single block [start_time, end_time]
        - 'day'     : split into daily blocks
        - 'month'   : split into calendar months
        - 'quarter' : split into calendar quarters
        - 'year'    : split into calendar years
        - 'season'  : split into meteorological seasons (MAM, JJA, SON, DJF)
    inclusive_end_time: bool
        The default is True.
        If False, if the last block end_time is equal to input end_time, such block is removed.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 2) with dtype datetime64[s], where each row is [block_start, block_end].

    """
    freq = check_freq(freq)
    if freq == "none":
        return np.array([[start_time, end_time]], dtype="datetime64[s]")

    # Mapping from our custom freq to pandas frequency codes
    freq_map = {
        "hour": "h",
        "day": "d",
        "month": "M",
        "quarter": "Q",
        "year": "Y",
        "season": "Q-FEB",  # seasons DJF, MAM, JJA, SON
    }

    # Define periods
    periods = pd.period_range(start=start_time, end=end_time, freq=freq_map[freq])

    # Create time blocks
    blocks = []
    for period in periods:
        start = period.start_time.to_datetime64().astype("datetime64[s]")
        if freq == "quarter":
            end = period.end_time.floor("s").to_datetime64().astype("datetime64[s]")
        else:
            end = period.end_time.to_datetime64().astype("datetime64[s]")
        blocks.append([start, end])
    blocks = np.array(blocks, dtype="datetime64[s]")

    if not inclusive_end_time and len(blocks) > 0 and blocks[-1, 0] == end_time:
        blocks = blocks[:-1]
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


def identify_time_partitions(start_times, end_times, freq: str) -> list[dict]:
    """Identify the set of time blocks covered by files.

    The result is a minimal, sorted, and unique set of time partitions.
    'start_times' and end_times can be derived using get_start_end_time_from_filepaths.

    Parameters
    ----------
    start_times : numpy.ndarray of datetime64[s]
        Array of inclusive start times for each file.
    end_times : numpy.ndarray of datetime64[s]
        Array of inclusive end times for each file.
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
    # Define files time coverage
    start_time, end_time = start_times.min(), end_times.max()

    # Compute candidate time blocks
    blocks = generate_time_blocks(start_time, end_time, freq=freq)

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

    start_times, end_times = get_start_end_time_from_filepaths(filepaths)
    return identify_time_partitions(start_times=start_times, end_times=end_times, **strategy_options)


####----------------------------------------------------------------------------
#### Filepaths partitioning


def _map_files_to_blocks(files_start_time, files_end_time, filepaths, block_starts, block_ends):
    """Map each block start_time to list of overlapping filepaths."""
    # Use broadcasting to create a boolean matrix indicating which files cover which time block
    # Broadcasting: (n_files, n_blocks)
    mask = (files_start_time[:, None] <= block_ends[None, :]) & (files_end_time[:, None] >= block_starts[None, :])
    # Create a list with the a dictionary for each block
    filepaths = np.array(filepaths)
    results = []
    for i, (start, end) in enumerate(zip(block_starts, block_ends)):
        indices = np.where(mask[:, i])[0]
        if indices.size > 0:
            results.append(
                {"start_time": start, "end_time": end, "filepaths": filepaths[indices].tolist()},
            )
    return results


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
    if len(filepaths) == 0 or len(list_partitions) == 0:
        return []

    # Ensure sample_interval and accumulation_interval is numpy.timedelta64
    accumulation_interval = ensure_timedelta_seconds(accumulation_interval)
    sample_interval = ensure_timedelta_seconds(sample_interval)

    # Define offset on event_end_time
    offset = accumulation_interval if sample_interval != accumulation_interval else ensure_timedelta_seconds(0)

    # Retrieve file start_time and end_time
    files_start_time, files_end_time = get_start_end_time_from_filepaths(filepaths)

    # Retrieve partitions blocks start and end time arrays
    block_starts = np.array([p["start_time"] for p in list_partitions])
    block_ends = np.array([p["end_time"] for p in list_partitions])

    # Add optional offset for resampling
    # TODO: expanding partition time should be done only at L1 stage when resampling
    # In disdrodb, the time reported is time at the start of the accumulation period !
    # If sensors report time at the end of measurement interval, we might being reporting time
    #  with an inaccuracy equals to the sensor measurement interval.
    # We could correct for that at L0C stage already !
    block_ends = block_ends + offset

    # Retrieve partitioning info dictionaries
    event_info = _map_files_to_blocks(files_start_time, files_end_time, filepaths, block_starts, block_ends)
    return event_info


def get_files_per_time_block(filepaths, freq="day", tolerance_seconds=120):
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
    files_start_time = files_start_time - np.array(tolerance_seconds, dtype="m8[s]")
    files_end_time = files_end_time + np.array(tolerance_seconds, dtype="m8[s]")

    # Identify candidate blocks
    list_partitions = identify_time_partitions(
        start_times=files_start_time,
        end_times=files_end_time,
        freq=freq,
    )
    block_starts = np.array([b["start_time"] for b in list_partitions])
    block_ends = np.array([b["end_time"] for b in list_partitions])

    # Map filepaths to corresponding time blocks
    list_dicts = _map_files_to_blocks(files_start_time, files_end_time, filepaths, block_starts, block_ends)

    # Create a dictionary of format {block_start_time: filepaths}
    dict_blocks = {str(d["start_time"]): d["filepaths"] for d in list_dicts}
    return dict_blocks
