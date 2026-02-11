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
"""This module contains utilities related to the processing of temporal dataset."""

import logging
import numbers
import re

import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.utils.logger import log_info, log_warning
from disdrodb.utils.xarray import define_fill_value_dictionary

logger = logging.getLogger(__name__)


####------------------------------------------------------------------------------------.
#### Sampling Interval Acronyms


def seconds_to_temporal_resolution(seconds):
    """Convert a duration in seconds to a readable string format (e.g., "1H30", "1D2H").

    Parameters
    ----------
    seconds: int
        The time duration in seconds.

    Returns
    -------
    str:
        The duration as a string in a format like "30S", "1MIN30S", "1H30MIN", or "1D2H".
    """
    timedelta = pd.Timedelta(seconds=seconds)
    components = timedelta.components

    parts = []
    if components.days > 0:
        parts.append(f"{components.days}D")
    if components.hours > 0:
        parts.append(f"{components.hours}H")
    if components.minutes > 0:
        parts.append(f"{components.minutes}MIN")
    if components.seconds > 0:
        parts.append(f"{components.seconds}S")
    temporal_resolution = "".join(parts)
    return temporal_resolution


def get_sampling_information(temporal_resolution):
    """
    Extract resampling information from the temporal_resolution string.

    Parameters
    ----------
    temporal_resolution: str
      A string representing the product temporal resolution: e.g., "1H30MIN", "ROLL1H30MIN".

    Returns
    -------
    sample_interval_seconds, rolling: tuple
        Sample_interval in seconds and whether rolling is enabled.
    """
    rolling = temporal_resolution.startswith("ROLL")
    if rolling:
        temporal_resolution = temporal_resolution[4:]  # Remove "ROLL"

    # Allowed pattern: one or more occurrences of "<number><unit>"
    # where unit is exactly one of D, H, MIN, or S.
    # Examples: 1H, 30MIN, 2D, 45S, and any concatenation like 1H30MIN.
    pattern = r"^(\d+(?:D|H|MIN|S))+$"

    # Check if the entire string matches the pattern
    if not re.match(pattern, temporal_resolution):
        raise ValueError(
            f"Invalid temporal resolution '{temporal_resolution}'. "
            "Must be composed of one or more <number><unit> groups, where unit is D, H, MIN, or S.",
        )

    # Regular expression to match duration components and extract all (value, unit) pairs
    pattern = r"(\d+)(D|H|MIN|S)"
    matches = re.findall(pattern, temporal_resolution)

    # Conversion factors for each unit
    unit_to_seconds = {
        "D": 86400,  # Seconds in a day
        "H": 3600,  # Seconds in an hour
        "MIN": 60,  # Seconds in a minute
        "S": 1,  # Seconds in a second
    }

    # Parse matches and calculate total seconds
    sample_interval = 0
    for value, unit in matches:
        value = int(value)
        if unit in unit_to_seconds:
            sample_interval += value * unit_to_seconds[unit]
    return sample_interval, rolling


def temporal_resolution_to_seconds(temporal_resolution):
    """
    Extract the measurement interval in seconds from the temporal resolution string.

    Parameters
    ----------
    temporal_resolution: str
      A string representing the product measurement interval: e.g., "1H30MIN", "ROLL1H30MIN".

    Returns
    -------
    int
        Duration in seconds.
    """
    seconds, _ = get_sampling_information(temporal_resolution)
    return seconds


####----------------------------------------------------------------------------.
#### File start and end time utilities
def get_dataframe_start_end_time(df: pd.DataFrame, time_column="time"):
    """Retrieves dataframe starting and ending time.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    time_column: str
        Name of the time column.
        The default is "time".
        The column must be of type datetime.

    Returns
    -------
    (start_time, end_time): tuple
        File start and end time of type pandas.Timestamp.

    """
    starting_time = pd.to_datetime(df[time_column].iloc[0])
    ending_time = pd.to_datetime(df[time_column].iloc[-1])
    return (starting_time, ending_time)


def get_dataset_start_end_time(ds: xr.Dataset, time_dim="time"):
    """Retrieves dataset starting and ending time.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset
    time_dim: str
        Name of the time dimension.
        The default is "time".

    Returns
    -------
    (start_time, end_time): tuple
        File start and end time of type pandas.Timestamp.

    """
    starting_time = pd.to_datetime(ds[time_dim].to_numpy()[0])
    ending_time = pd.to_datetime(ds[time_dim].to_numpy()[-1])
    return (starting_time, ending_time)


def get_file_start_end_time(obj, time="time"):
    """Retrieves object starting and ending time.

    Parameters
    ----------
    obj : xarray.Dataset or pandas.DataFrame
        Input object with time dimension or column respectively.
    time: str
        Name of the time dimension or column.
        The default is "time".

    Returns
    -------
    (start_time, end_time): tuple
        File start and end time of type pandas.Timestamp.

    """
    if isinstance(obj, xr.Dataset):
        return get_dataset_start_end_time(obj, time_dim=time)
    if isinstance(obj, pd.DataFrame):
        return get_dataframe_start_end_time(obj, time_column=time)
    raise TypeError("Expecting a xarray Dataset or a pandas Dataframe object.")


####------------------------------------------------------------------------------------.
#### Xarray utilities


def ensure_sorted_by_time(obj, time="time"):
    """Ensure a xarray object or pandas Dataframe is sorted by time."""
    # Check sorted by time and sort if necessary
    is_sorted = np.all(np.diff(obj[time].to_numpy().astype(int)) > 0)
    if not is_sorted:
        if isinstance(obj, pd.DataFrame):
            return obj.sort_values(by="time")
        # Else xarray DataArray or Dataset
        obj = obj.sortby("time")
    return obj


def _check_time_sorted(ds, time_dim):
    """Ensure the xarray.Dataset is sorted."""
    time_diff = np.diff(ds[time_dim].to_numpy().astype(int))
    if np.any(time_diff == 0):
        raise ValueError(f"In the {time_dim} dimension there are duplicated timesteps !")
    if not np.all(time_diff > 0):
        print(f"The {time_dim} dimension was not sorted. Sorting it now !")
        ds = ds.sortby(time_dim)
    return ds


def regularize_dataset(
    xr_obj,
    freq: str,
    time_dim: str = "time",
    method: str | None = None,
    fill_value=None,
    start_time=None,
    end_time=None,
):
    """Regularize a xarray object across time dimension with uniform resolution.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray
        xarray object with time dimension.
    time_dim : str, optional
        The time dimension in the xarray object. The default value is ``"time"``.
    freq : str
        The ``freq`` string to pass to `pd.date_range()` to define the new time coordinates.
        Examples: ``freq="2min"``.
    method : str, optional
        Method to use for filling missing timesteps.
        If ``None``, fill with ``fill_value``. The default value is ``None``.
        For other possible methods, see xarray.Dataset.reindex()`.
    fill_value : float or dict, optional
        Fill value to fill missing timesteps.
        If not specified, for float variables it uses ``dtypes.NA`` while for
        for integers variables it uses the maximum allowed integer value or,
        in case of undecoded variables, the ``_FillValue`` DataArray attribute.

    Returns
    -------
    ds_reindexed : xarray.Dataset
        Regularized dataset.

    """
    attrs = xr_obj.attrs.copy()
    xr_obj = _check_time_sorted(xr_obj, time_dim=time_dim)

    # Define start time and end_time
    start, end = get_dataset_start_end_time(xr_obj, time_dim=time_dim)
    if start_time is None:
        start_time = start
    if end_time is None:
        end_time = end
    xr_obj = xr_obj.sel({time_dim: slice(start_time, end_time)})

    # Define new time index
    new_time_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq=freq,
    )
    # Check all existing timesteps are within the new time index
    # - Otherwise raise error because it means that the desired frequency is not compatible
    idx_missing = np.where(~np.isin(xr_obj[time_dim].data, new_time_index))[0]
    if len(idx_missing) > 0:
        not_included_timesteps = xr_obj[time_dim].data[idx_missing].astype("M8[s]")
        raise ValueError(f"With freq='{freq}', the following timesteps would be dropped: {not_included_timesteps}")

    # Define fill_value dictionary
    if fill_value is None:
        fill_value = define_fill_value_dictionary(xr_obj)

    # Regularize dataset and fill with NA values
    xr_obj = xr_obj.reindex(
        {time_dim: new_time_index},
        method=method,  # do not fill gaps
        # tolerance=tolerance,  # mismatch in seconds
        fill_value=fill_value,
    )

    # Ensure attributes are preserved
    xr_obj.attrs = attrs
    return xr_obj


####------------------------------------------
#### Interval utilities


def ensure_sample_interval_in_seconds(sample_interval):  # noqa: PLR0911
    """
    Ensure the sample interval is in seconds.

    Parameters
    ----------
    sample_interval : int, numpy.ndarray, xarray.DataArray, or numpy.timedelta64
        The sample interval to be converted to seconds.
        It can be:

        - An integer representing the interval in seconds.
        - A numpy array or xarray DataArray of integers representing intervals in seconds.
        - A numpy.timedelta64 object representing the interval.
        - A numpy array or xarray DataArray of numpy.timedelta64 objects representing intervals.


    Returns
    -------
    int, numpy.ndarray, or xarray.DataArray
        The sample interval converted to seconds. The return type matches the input type:

        - If the input is an integer, the output is an integer.
        - If the input is a numpy array, the output is a numpy array of integers (unless NaN is present)
        - If the input is an xarray DataArray, the output is an xarray DataArray of integers (unless NaN is present).


    """
    # Deal with timedelta objects
    if isinstance(sample_interval, np.timedelta64):
        return (sample_interval.astype("m8[s]") / np.timedelta64(1, "s")).astype(int)
        # return sample_interval.astype("m8[s]").astype(int)

    # Deal with scalar pure integer types (Python int or numpy int32/int64/etc.)
    # --> ATTENTION: this also include np.timedelta64 objects !
    if isinstance(sample_interval, numbers.Integral):
        return sample_interval

    # Deal with numpy or xarray arrays of integer types
    if isinstance(sample_interval, (np.ndarray, xr.DataArray)) and np.issubdtype(sample_interval.dtype, int):
        return sample_interval

    # Deal with scalar floats that are actually integers (e.g. 1.0, np.float64(3.0))
    if isinstance(sample_interval, numbers.Real):
        if float(sample_interval).is_integer():
            # Cast back to int seconds
            return int(sample_interval)
        raise TypeError(f"sample_interval floats must be whole numbers of seconds, got {sample_interval}")

    # Deal with timedelta64 numpy arrays
    if isinstance(sample_interval, np.ndarray) and np.issubdtype(sample_interval.dtype, np.timedelta64):
        is_nat = np.isnat(sample_interval)
        if np.any(is_nat):
            sample_interval = sample_interval.astype("timedelta64[s]").astype(float)
            sample_interval[is_nat] = np.nan
            return sample_interval
        return sample_interval.astype("timedelta64[s]").astype(int)
    # Deal with timedelta64 xarray arrays
    if isinstance(sample_interval, xr.DataArray) and np.issubdtype(sample_interval.dtype, np.timedelta64):
        sample_interval = sample_interval.copy()
        is_nat = np.isnat(sample_interval)
        if np.any(is_nat):
            sample_interval_array = sample_interval.data.astype("timedelta64[s]").astype(float)
            sample_interval_array[is_nat] = np.nan
            sample_interval.data = sample_interval_array
            return sample_interval
        sample_interval_array = sample_interval.data.astype("timedelta64[s]").astype(int)
        sample_interval.data = sample_interval_array
        return sample_interval

    # Deal with numpy array of floats that are all integer-valued (with optionally some NaN)
    if isinstance(sample_interval, np.ndarray) and np.issubdtype(sample_interval.dtype, np.floating):
        mask_nan = np.isnan(sample_interval)
        if mask_nan.any():
            # Check non-NaN entries are whole numbers
            nonnan = sample_interval[~mask_nan]
            if not np.allclose(nonnan, np.rint(nonnan)):
                raise TypeError("Float array sample_interval must contain only whole numbers or NaN.")
            # Leave as float array so NaNs are preserved
            return sample_interval
        # No NaNs: can safely cast to integer dtype
        if not np.allclose(sample_interval, np.rint(sample_interval)):
            raise TypeError("Float array sample_interval must contain only whole numbers.")
        return sample_interval.astype(int)

    # Deal with xarray.DataArray of floats that are all integer-valued (with optionally some NaN)
    if isinstance(sample_interval, xr.DataArray) and np.issubdtype(sample_interval.dtype, np.floating):
        arr = sample_interval.copy()
        data = arr.data
        mask_nan = np.isnan(data)
        if mask_nan.any():
            nonnan = data[~mask_nan]
            if not np.allclose(nonnan, np.rint(nonnan)):
                raise TypeError("Float DataArray sample_interval must contain only whole numbers or NaN.")
            # return as float DataArray so NaNs stay
            return arr
        if not np.allclose(data, np.rint(data)):
            raise TypeError("Float DataArray sample_interval must contain only whole numbers.")
        arr.data = data.astype(int)
        return arr

    raise TypeError(
        "sample_interval must be an integer value or array, or numpy.ndarray / xarray.DataArray with type timedelta64.",
    )


def ensure_timedelta_seconds(interval):
    """Return an a scalar value/array in seconds or timedelta object as numpy.timedelta64 in seconds."""
    if isinstance(interval, (xr.DataArray, np.ndarray)):
        return ensure_sample_interval_in_seconds(interval).astype("m8[s]")
    return np.array(ensure_sample_interval_in_seconds(interval), dtype="m8[s]")


####------------------------------------------
#### Sample Interval Utilities


def infer_sample_interval(ds, robust=False, verbose=False, logger=None):
    """Infer the sample interval of a dataset.

    Duplicated timesteps are removed before inferring the sample interval.

    NOTE: This function is used only for the reader preparation.
    """
    # Check sorted by time and sort if necessary
    ds = ensure_sorted_by_time(ds)

    # Retrieve timesteps
    # - Remove duplicate timesteps
    timesteps = np.unique(ds["time"].data)

    # Calculate number of timesteps
    n_timesteps = len(timesteps)

    # Calculate time differences in seconds
    deltadt = np.diff(timesteps).astype("timedelta64[s]").astype(int)

    # Round each delta to the nearest multiple of 5 (because the smallest possible sample interval is 10 s)
    # Example: for sample_interval = 10, deltat values like 8, 9, 11, 12 become 10 ...
    # Example: for sample_interval = 10, deltat values like 6, 7 or 13, 14 become respectively 5 and 15 ...
    # Example: for sample_interval = 30, deltat values like 28,29,30,31,32 deltat  become 30 ...
    # Example: for sample_interval = 30, deltat values like 26, 27 or 33, 34 become respectively 25 and 35 ...
    # --> Need other rounding after having identified the most frequent sample interval to coerce such values to 30
    min_sample_interval = 10
    min_half_sample_interval = min_sample_interval / 2
    deltadt = np.round(deltadt / min_half_sample_interval) * min_half_sample_interval

    # Identify unique time intervals and their occurrences
    unique_deltas, counts = np.unique(deltadt, return_counts=True)

    # Determine the most frequent time interval (mode)
    most_frequent_delta_idx = np.argmax(counts)
    sample_interval = unique_deltas[most_frequent_delta_idx]

    # Reround deltadt once knowing the sample interval
    # - If sample interval is 10: all values between 6 and 14 are rounded to 10, below 6 to 0, above 14 to 20
    # - If sample interval is 30: all values between 16 and 44 are rounded to 30, below 16 to 0, above 44 to 20
    deltadt = np.round(deltadt / min_sample_interval) * min_sample_interval

    # Identify unique time intervals and their occurrences
    unique_deltas, counts = np.unique(deltadt, return_counts=True)
    fractions = np.round(counts / len(deltadt) * 100, 2)

    # Determine the most frequent time interval (mode)
    most_frequent_delta_idx = np.argmax(counts)
    sample_interval = unique_deltas[most_frequent_delta_idx]
    sample_interval_fraction = fractions[most_frequent_delta_idx]

    # Inform about irregular sampling
    unexpected_intervals = unique_deltas[unique_deltas != sample_interval]
    unexpected_intervals_counts = counts[unique_deltas != sample_interval]
    unexpected_intervals_fractions = fractions[unique_deltas != sample_interval]
    if verbose and len(unexpected_intervals) > 0:
        msg = "Non-unique interval detected."
        log_info(logger=logger, msg=msg, verbose=verbose)
        for interval, count, fraction in zip(
            unexpected_intervals,
            unexpected_intervals_counts,
            unexpected_intervals_fractions,
            strict=True,
        ):
            msg = f"--> Interval: {interval} seconds, Occurrence: {count}, Frequency: {fraction} %"
            log_info(logger=logger, msg=msg, verbose=verbose)

    # Perform checks
    # - Raise error if negative or zero time intervals are presents
    # - If robust = False, still return the estimated sample_interval
    if robust and np.any(deltadt == 0):
        raise ValueError("Likely presence of duplicated timesteps.")

    if robust and len(unexpected_intervals) > 0:
        raise ValueError("Not unique sampling interval.")

    ###-------------------------------------------------------------------------.
    ### Display informative messages
    # - Log a warning if estimated sample interval has frequency less than 60 %
    sample_interval_fraction_threshold = 60
    msg = (
        f"The most frequent sampling interval ({sample_interval} s) "
        + f"has a frequency lower than {sample_interval_fraction_threshold}%: {sample_interval_fraction} %. "
        + f"(Total number of timesteps: {n_timesteps})"
    )
    if sample_interval_fraction < sample_interval_fraction_threshold:
        log_warning(logger=logger, msg=msg, verbose=verbose)

    # - Log a warning if an unexpected interval has frequency larger than 20 percent
    frequent_unexpected_intervals = unexpected_intervals[unexpected_intervals_fractions > 20]
    if len(frequent_unexpected_intervals) != 0:
        frequent_unexpected_intervals_str = ", ".join(
            f"{interval} seconds" for interval in frequent_unexpected_intervals
        )
        msg = (
            "The following unexpected intervals have a frequency "
            + f"greater than 20%: {frequent_unexpected_intervals_str}. "
            + f"(Total number of timesteps: {n_timesteps})"
        )
        log_warning(logger=logger, msg=msg, verbose=verbose)
    return int(sample_interval)
