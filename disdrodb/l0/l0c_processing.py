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
"""Functions to process DISDRODB L0B files into DISDRODB L0C netCDF files."""
import logging

import numpy as np
import pandas as pd

from disdrodb.api.checks import check_measurement_intervals
from disdrodb.api.info import get_start_end_time_from_filepaths
from disdrodb.l1.resampling import add_sample_interval
from disdrodb.utils.logger import log_warning  # , log_info
from disdrodb.utils.time import (
    ensure_sorted_by_time,
    regularize_timesteps,
)

logger = logging.getLogger(__name__)


TOLERANCE_SECONDS = 120


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


def retrieve_possible_measurement_intervals(metadata):
    """Retrieve list of possible measurements intervals."""
    measurement_intervals = metadata.get("measurement_interval", [])
    return check_measurement_intervals(measurement_intervals)


def drop_timesteps_with_invalid_sample_interval(ds, measurement_intervals, verbose=True, logger=None):
    """Drop timesteps with unexpected sample intervals."""
    # TODO
    # - correct logged sample_interval for trailing seconds. Example (58,59,61,62) converted to 60 s ?
    # - Need to know more how Parsivel software computes sample_interval variable ...

    # Retrieve logged sample_interval
    sample_interval = ds["sample_interval"].compute().data
    timesteps = ds["time"].compute().data
    is_valid_sample_interval = np.isin(sample_interval.data, measurement_intervals)
    indices_invalid_sample_interval = np.where(~is_valid_sample_interval)[0]
    if len(indices_invalid_sample_interval) > 0:
        # Log information for each invalid timestep
        invalid_timesteps = pd.to_datetime(timesteps[indices_invalid_sample_interval]).strftime("%Y-%m-%d %H:%M:%S")
        invalid_sample_intervals = sample_interval[indices_invalid_sample_interval]
        for tt, ss in zip(invalid_timesteps, invalid_sample_intervals):
            msg = f"Unexpected sampling interval ({ss} s) at {tt}. The measurement has been dropped."
            log_warning(logger=logger, msg=msg, verbose=verbose)
        # Remove timesteps with invalid sample intervals
        indices_valid_sample_interval = np.where(is_valid_sample_interval)[0]
        ds = ds.isel(time=indices_valid_sample_interval)
    return ds


def split_dataset_by_sampling_intervals(ds, measurement_intervals, min_sample_interval=10, min_block_size=5):
    """
    Split a dataset into subsets where each subset has a consistent sampling interval.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset with a 'time' dimension.
    measurement_intervals : list or array-like
        A list of possible primary sampling intervals (in seconds) that the dataset might have.
    min_sample_interval : int, optional
        The minimum expected sampling interval in seconds. Defaults to 10s.
    min_block_size : float, optional
        The minimum number of timesteps with a given sampling interval to be considered.
        Otherwise such portion of data is discarded !
        Defaults to 5 timesteps.

    Returns
    -------
    dict
        A dictionary where keys are the identified sampling intervals (in seconds),
        and values are xarray.Datasets containing only data from those intervals.
    """
    # Define array of possible measurement intervals
    measurement_intervals = np.array(measurement_intervals)

    # If a single measurement interval expected, return dictionary with input dataset
    if len(measurement_intervals) == 1:
        dict_ds = {measurement_intervals[0]: ds}
        return dict_ds

    # Check sorted by time and sort if necessary
    ds = ensure_sorted_by_time(ds)

    # Calculate time differences in seconds
    deltadt = np.diff(ds["time"].data).astype("timedelta64[s]").astype(int)

    # Round each delta to the nearest multiple of 5 (because the smallest possible sample interval is 10 s)
    # - This account for possible trailing seconds of the logger
    # Example: for sample_interval = 10, deltat values like 8, 9, 11, 12 become 10 ...
    # Example: for sample_interval = 10, deltat values like 6, 7 or 13, 14 become respectively 5 and 15 ...
    # Example: for sample_interval = 30, deltat values like 28,29,30,31,32 deltat  become 30 ...
    # Example: for sample_interval = 30, deltat values like 26, 27 or 33, 34 become respectively 25 and 35 ...
    min_half_sample_interval = min_sample_interval / 2
    deltadt = np.round(deltadt / min_half_sample_interval) * min_half_sample_interval

    # Map each delta to one of the possible_measurement_intervals if exact match, otherwise np.nan
    mapped_intervals = np.where(np.isin(deltadt, measurement_intervals), deltadt, np.nan)
    if np.all(np.isnan(mapped_intervals)):
        raise ValueError("Impossible to identify timesteps with expected sampling intervals.")

    # Infill np.nan values by using neighbor intervals
    # Forward fill
    for i in range(1, len(mapped_intervals)):
        if np.isnan(mapped_intervals[i]):
            mapped_intervals[i] = mapped_intervals[i - 1]

    # Backward fill (in case the first entries were np.nan)
    for i in range(len(mapped_intervals) - 2, -1, -1):
        if np.isnan(mapped_intervals[i]):
            mapped_intervals[i] = mapped_intervals[i + 1]

    # Now all intervals are assigned to one of the possible measurement_intervals.
    # Identify boundaries where interval changes
    change_points = np.where(mapped_intervals[:-1] != mapped_intervals[1:])[0] + 1

    # Split ds into segments according to change_points
    segments = np.split(np.arange(ds.sizes["time"]), change_points)

    # Remove segments with less than 10 points
    segments = [seg for seg in segments if len(seg) >= min_block_size]
    if len(segments) == 0:
        raise ValueError(
            f"No blocks of {min_block_size} consecutive timesteps with constant sampling interval are available.",
        )

    # Define dataset indices for each sampling interva
    dict_sampling_interval_indices = {}
    for seg in segments:
        # Define the assumed sampling interval of such segment
        start_idx = seg[0]
        segment_sampling_interval = int(mapped_intervals[start_idx])
        if segment_sampling_interval not in dict_sampling_interval_indices:
            dict_sampling_interval_indices[segment_sampling_interval] = [seg]
        else:
            dict_sampling_interval_indices[segment_sampling_interval].append(seg)
    dict_sampling_interval_indices = {
        k: np.concatenate(list_indices) for k, list_indices in dict_sampling_interval_indices.items()
    }

    # Define dictionary of datasets
    dict_ds = {k: ds.isel(time=indices) for k, indices in dict_sampling_interval_indices.items()}
    return dict_ds


def has_same_value_over_time(da):
    """
    Check if a DataArray has the same value over all timesteps, considering NaNs as equal.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to check. Must have a 'time' dimension.

    Returns
    -------
    bool
        True if the values are the same (or NaN in the same positions) across all timesteps,
        False otherwise.
    """
    # Select the first timestep
    da_first = da.isel(time=0)

    # Create a boolean array that identifies where values are equal or both NaN
    equal_or_nan = (da == da_first) | (da.isnull() & da_first.isnull())  # noqa: PD003

    # Check if all values match this condition across all dimensions
    return bool(equal_or_nan.all().item())


def remove_duplicated_timesteps(ds, ensure_variables_equality=True, logger=None, verbose=True):
    """Removes duplicated timesteps from a xarray dataset."""
    # Check for duplicated timesteps
    timesteps, counts = np.unique(ds["time"].data, return_counts=True)
    duplicated_timesteps = timesteps[counts > 1]

    # If no duplicated timesteps, returns dataset as is
    if len(duplicated_timesteps) == 0:
        return ds

    # If there are duplicated timesteps
    # - First check for variables equality
    # - Keep first occurrence of duplicated timesteps if values are equals
    # - Drop duplicated timesteps where values are different
    different_duplicated_timesteps = []
    equal_duplicated_timesteps = []
    for t in duplicated_timesteps:
        # Select dataset at given duplicated timestep
        ds_duplicated = ds.sel(time=t)
        n_t = len(ds_duplicated["time"])

        # Check raw_drop_number equality
        if not has_same_value_over_time(ds_duplicated["raw_drop_number"]):
            different_duplicated_timesteps.append(t)
            msg = (
                f"Presence of {n_t} duplicated timesteps at {t}."
                "They have different 'raw_drop_number' values. These timesteps are dropped."
            )
            log_warning(logger=logger, msg=msg, verbose=verbose)

        # Check other variables equality
        other_variables_to_check = [v for v in ds.data_vars if v != "raw_drop_number"]
        variables_with_different_values = [
            var for var in other_variables_to_check if not has_same_value_over_time(ds_duplicated[var])
        ]
        if len(variables_with_different_values) > 0:
            msg = (
                f"Presence of {n_t} duplicated timesteps at {t}."
                f"The duplicated timesteps have different values in variables {variables_with_different_values}. "
            )
            if ensure_variables_equality:
                different_duplicated_timesteps.append(t)
                msg = msg + "These timesteps are dropped."
            else:
                equal_duplicated_timesteps.append(t)
                msg = msg + (
                    "These timesteps are not dropped because 'raw_drop_number' values are equals."
                    "'ensure_variables_equality' is False."
                )
            log_warning(logger=logger, msg=msg, verbose=verbose)
        else:
            equal_duplicated_timesteps.append(t)

    # Ensure single occurrence of duplicated timesteps
    equal_duplicated_timesteps = np.unique(equal_duplicated_timesteps)
    different_duplicated_timesteps = np.unique(different_duplicated_timesteps)

    # - Keep first occurrence of equal_duplicated_timesteps
    if len(equal_duplicated_timesteps) > 0:
        indices_to_drop = [np.where(ds["time"] == t)[0][1:] for t in equal_duplicated_timesteps]
        indices_to_drop = np.concatenate(indices_to_drop)
        # Keep only indices not in indices_to_drop
        mask = ~np.isin(np.arange(ds["time"].size), indices_to_drop)
        ds = ds.isel(time=np.where(mask)[0])

    # - Drop different_duplicated_timesteps
    if len(different_duplicated_timesteps) > 0:
        mask = np.isin(ds["time"], different_duplicated_timesteps, invert=True)
        ds = ds.isel(time=np.where(mask)[0])

    return ds


def check_timesteps_regularity(ds, sample_interval, verbose=False, logger=None):
    """Check for the regularity of timesteps."""
    # Check sorted by time and sort if necessary
    ds = ensure_sorted_by_time(ds)

    # Calculate number of timesteps
    n = len(ds["time"].data)

    # Calculate time differences in seconds
    deltadt = np.diff(ds["time"].data).astype("timedelta64[s]").astype(int)

    # Identify unique time intervals and their occurrences
    unique_deltadt, counts = np.unique(deltadt, return_counts=True)

    # Determine the most frequent time interval (mode)
    most_frequent_deltadt_idx = np.argmax(counts)
    most_frequent_deltadt = unique_deltadt[most_frequent_deltadt_idx]

    # Count fraction occurrence of deltadt
    fractions = np.round(counts / len(deltadt) * 100, 2)

    # Compute stats about expected deltadt
    sample_interval_counts = counts[unique_deltadt == sample_interval].item()
    sample_interval_fraction = fractions[unique_deltadt == sample_interval].item()

    # Compute stats about most frequent deltadt
    most_frequent_deltadt_counts = counts[unique_deltadt == most_frequent_deltadt].item()
    most_frequent_deltadt_fraction = fractions[unique_deltadt == most_frequent_deltadt].item()

    # Compute stats about unexpected deltadt
    unexpected_intervals = unique_deltadt[unique_deltadt != sample_interval]
    unexpected_intervals_counts = counts[unique_deltadt != sample_interval]
    unexpected_intervals_fractions = fractions[unique_deltadt != sample_interval]
    frequent_unexpected_intervals = unexpected_intervals[unexpected_intervals_fractions > 5]

    # Report warning if the samplin_interval deltadt occurs less often than 60 % of times
    # -> TODO: maybe only report in stations where the disdro does not log only data when rainy
    if sample_interval_fraction < 60:
        msg = (
            f"The expected (sampling) interval between observations occurs only "
            f"{sample_interval_counts}/{n} times ({sample_interval_fraction} %)."
        )

    # Report warning if a deltadt occurs more often then the sampling interval
    if most_frequent_deltadt != sample_interval:
        msg = (
            f"The most frequent time interval between observations is {most_frequent_deltadt} s "
            f"(occurs {most_frequent_deltadt_counts}/{n} times) ({most_frequent_deltadt_fraction}%) "
            f"although the expected (sampling) interval is {sample_interval} s "
            f"and occurs {sample_interval_counts}/{n} times ({sample_interval_fraction}%)."
        )
        log_warning(logger=logger, msg=msg, verbose=verbose)

    # Report with a warning all unexpected deltadt with frequency larger than 5 %
    if len(frequent_unexpected_intervals) > 0:
        msg_parts = ["The following unexpected intervals occur frequently:"]
        for interval in frequent_unexpected_intervals:
            c = unexpected_intervals_counts[unexpected_intervals == interval].item()
            f = unexpected_intervals_fractions[unexpected_intervals == interval].item()
            msg_parts.append(f" {interval} ({f}%) ({c}/{n}) | ")
        msg = " ".join(msg_parts)

        msg = "The following time intervals between observations occurs often: "
        for interval in frequent_unexpected_intervals:
            c = unexpected_intervals_counts[unexpected_intervals == interval].item()
            f = unexpected_intervals_fractions[unexpected_intervals == interval].item()
            msg = msg + f"{interval} s ({f}%) ({c}/{n})"
        log_warning(logger=logger, msg=msg, verbose=verbose)
    return ds


def finalize_l0c_dataset(ds, sample_interval, start_day, end_day, verbose=True, logger=None):
    """Finalize a L0C dataset with unique sampling interval.

    It adds the sampling_interval coordinate and it regularizes
    the timesteps for trailing seconds.
    """
    # Add sample interval as coordinate
    ds = add_sample_interval(ds, sample_interval=sample_interval)

    # Regularize timesteps (for trailing seconds)
    ds = regularize_timesteps(
        ds,
        sample_interval=sample_interval,
        robust=False,  # if True, raise error if an error occur during regularization
        add_quality_flag=True,
        verbose=verbose,
        logger=logger,
    )

    # Performs checks about timesteps regularity
    ds = check_timesteps_regularity(ds=ds, sample_interval=sample_interval, verbose=verbose, logger=logger)

    # Slice for requested day
    ds = ds.sel({"time": slice(start_day, end_day)})
    return ds


def create_daily_file(day, filepaths, measurement_intervals, ensure_variables_equality=True, logger=None, verbose=True):
    """
    Create a daily file by merging and processing data from multiple filepaths.

    Parameters
    ----------
    day : str or numpy.datetime64
        The day for which the daily file is to be created.
        Should be in a format that can be converted to numpy.datetime64.
    filepaths : list of str
        List of filepaths to the data files to be processed.

    Returns
    -------
    xarray.Dataset
        The processed dataset containing data for the specified day.

    Raises
    ------
    ValueError
        If less than 5 timesteps are available for the specified day.

    Notes
    -----
    - The function adds a tolerance for searching timesteps
    before and after 00:00 to account for imprecise logging times.
    - It checks that duplicated timesteps have the same raw drop number values.
    - The function infers the time integration sample interval and
    regularizes timesteps to handle trailing seconds.
    - The data is loaded into memory and connections to source files
    are closed before returning the dataset.
    """
    import xarray as xr  # Load in each process when function is called !

    # ---------------------------------------------------------------------------------------.
    # Define start day and end of day
    start_day = np.array(day).astype("M8[D]")
    end_day = start_day + np.array(1, dtype="m8[D]") - np.array(1, dtype="m8[s]")  # avoid 00:00 of next day !

    # Add tolerance for searching timesteps before and after 00:00 to account for imprecise logging time
    # - Example: timestep 23:59:30 that should be 00.00 goes into the next day ...
    start_day_tol = start_day - np.array(TOLERANCE_SECONDS, dtype="m8[s]")
    end_day_tol = end_day + np.array(TOLERANCE_SECONDS, dtype="m8[s]")

    # ---------------------------------------------------------------------------------------.
    # Open files with data within the provided day and concatenate them
    # list_ds = [xr.open_dataset(filepath, chunks={}).sel({"time": slice(start_day_tol, end_day_tol)})
    # for filepath in filepaths]
    list_ds = [
        xr.open_dataset(filepath, decode_timedelta=False, chunks={}, cache=False).sortby("time")
        for filepath in filepaths
    ]
    list_ds = [ds.sel({"time": slice(start_day_tol, end_day_tol)}) for ds in list_ds]
    if len(list_ds) > 1:
        # Concatenate dataset
        # - If some variable are missing in one file, it is filled with NaN. This should not occur anyway.
        # - The resulting dataset can have duplicated timesteps !
        ds = xr.concat(list_ds, dim="time", join="outer", compat="no_conflicts", combine_attrs="override").sortby(
            "time",
        )
    else:
        ds = list_ds[0]

    # Compute data
    ds = ds.compute()

    # Close connection to source files
    _ = [ds.close() for ds in list_ds]
    ds.close()
    del list_ds

    # ---------------------------------------------------------------------------------------.
    # If sample interval is a dataset variable, drop timesteps with unexpected measurement intervals !
    if "sample_interval" in ds:
        ds = drop_timesteps_with_invalid_sample_interval(
            ds=ds,
            measurement_intervals=measurement_intervals,
            verbose=verbose,
            logger=logger,
        )

    # ---------------------------------------------------------------------------------------.
    # Remove duplicated timesteps
    ds = remove_duplicated_timesteps(
        ds,
        ensure_variables_equality=ensure_variables_equality,
        logger=logger,
        verbose=verbose,
    )

    # Raise error if less than 3 timesteps left
    n_timesteps = len(ds["time"])
    if n_timesteps < 3:
        raise ValueError(f"{n_timesteps} timesteps left after removing duplicated timesteps.")

    # ---------------------------------------------------------------------------------------.
    # Split dataset by sampling intervals
    dict_ds = split_dataset_by_sampling_intervals(
        ds=ds,
        measurement_intervals=measurement_intervals,
        min_sample_interval=10,
        min_block_size=5,
    )

    # Log a warning if two sampling intervals are present within a given day
    if len(dict_ds) > 1:
        occuring_sampling_intervals = list(dict_ds)
        msg = f"The dataset contains both sampling intervals {occuring_sampling_intervals}."
        log_warning(logger=logger, msg=msg, verbose=verbose)

    # ---------------------------------------------------------------------------------------.
    # Finalize L0C datasets
    # - Add sample_interval coordinate
    # - Regularize timesteps for trailing seconds
    dict_ds = {
        sample_interval: finalize_l0c_dataset(
            ds=ds,
            sample_interval=sample_interval,
            start_day=start_day,
            end_day=end_day,
            verbose=verbose,
            logger=logger,
        )
        for sample_interval, ds in dict_ds.items()
    }
    return dict_ds


# ---------------------------------------------------------------------------------------.
#### DEPRECATED CODE


# def copy_l0b_to_l0c_directory(filepath):
#     """Copy L0B file to L0C directory."""
#     import netCDF4

#     # Copy file
#     l0c_filepath = filepath.replace("L0B", "L0C")
#     _ = shutil.copy(filepath, l0c_filepath)

#     # Edit DISDRODB product attribute
#     with netCDF4.Dataset(l0c_filepath, mode="a") as nc_file:
#         # Modify the global attribute
#         nc_file.setncattr("disdrodb_product", "L0C")

# def find_isel_common_time(da1, da2):
#     """
#     Find the indices of common time steps between two data arrays.

#     Parameters
#     ----------
#     da1 : xarray.DataArray
#         The first data array with a time coordinate.
#     da2 : xarray.DataArray
#         The second data array with a time coordinate.

#     Returns
#     -------
#     da1_isel : numpy.ndarray
#         Indices of the common time steps in the first data array.
#     da2_isel : numpy.ndarray
#         Indices of the common time steps in the second data array.

#     Notes
#     -----
#     This function assumes that both input data arrays have a "time" coordinate.
#     The function finds the intersection of the time steps in both data arrays
#     and returns the indices of these common time steps for each data array.
#     """
#     intersecting_timesteps = np.intersect1d(da1["time"], da2["time"])
#     da1_isel = np.where(np.isin(da1["time"], intersecting_timesteps))[0]
#     da2_isel = np.where(np.isin(da2["time"], intersecting_timesteps))[0]
#     return da1_isel, da2_isel


# def check_same_raw_drop_number_values(list_ds, filepaths):
#     """
#     Check if the 'raw_drop_number' values are the same across multiple datasets.

#     This function compares the 'raw_drop_number' values of multiple datasets to ensure they are identical
#     at common timesteps.

#     If any discrepancies are found, a ValueError is raised indicating which files
#     have differing values.

#     Parameters
#     ----------
#     list_ds : list of xarray.Dataset
#         A list of xarray Datasets to be compared.
#     filepaths : list of str
#         A list of file paths corresponding to the datasets in `list_ds`.

#     Raises
#     ------
#     ValueError
#         If 'raw_drop_number' values differ at any common timestep between any two datasets.
#     """
#     # Retrieve variable to compare
#     list_drop_number = [ds["raw_drop_number"].compute() for ds in list_ds]
#     # Compare values
#     combos = list(itertools.combinations(range(len(list_drop_number)), 2))
#     for i, j in combos:
#         da1 = list_drop_number[i]
#         da2 = list_drop_number[j]
#         da1_isel, da2_isel = find_isel_common_time(da1=da1, da2=da2)
#         if not np.all(da1.isel(time=da1_isel).data == da2.isel(time=da2_isel).data):
#             file1 = filepaths[i]
#             file2 = filepaths[i]
#             msg = f"Duplicated timesteps have different values between file {file1} and {file2}"
#             raise ValueError(msg)
