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
import xarray as xr

from disdrodb.api.io import open_netcdf_files
from disdrodb.l0.l0b_processing import set_l0b_encodings
from disdrodb.l1.resampling import add_sample_interval
from disdrodb.utils.attrs import set_disdrodb_attrs
from disdrodb.utils.logger import log_warning, log_info
from disdrodb.utils.time import ensure_sorted_by_time

logger = logging.getLogger(__name__)

# L0C processing requires searching for data (per time blocks) into neighbouring files:
# - to account for possible trailing seconds in previous/next files
# - to get information if at the edges of the time blocks previous/next timesteps are available
# - to shift the time to ensure reported L0C time is the start of the measurement interval
TOLERANCE_SECONDS = 60 * 3

####---------------------------------------------------------------------------------
#### Measurement intervals


def drop_timesteps_with_invalid_sample_interval(ds, measurement_intervals, verbose=True, logger=None):
    """Drop timesteps with unexpected sample intervals."""
    sample_interval = ds["sample_interval"].to_numpy()
    timesteps = ds["time"].to_numpy()
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


def split_dataset_by_sampling_intervals(
    ds,
    measurement_intervals,
    min_sample_interval=10,
    min_block_size=5,
    time_is_end_interval=True,
):
    """
    Split a dataset into subsets where each subset has a consistent sampling interval.

    Notes
    -----
    - Does not modify timesteps (regularization is left to `regularize_timesteps`).
    - Assumes no duplicated timesteps in the dataset.
    - If only one measurement interval is specified, no timestep-diff checks are performed.
    - If multiple measurement intervals are specified:
        * Raises an error if *none* of the expected intervals appear.
        * Splits where interval changes.
    - Segments shorter than `min_block_size` are discarded.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset with a 'time' dimension.
    measurement_intervals : list or array-like
        A list of possible primary sampling intervals (in seconds) that the dataset might have.
    min_sample_interval : int, optional
        The minimum expected sampling interval in seconds. Defaults to 10s.
        This is used to deal with possible trailing seconds errors.
    min_block_size : float, optional
        The minimum number of timesteps with a given sampling interval to be considered.
        Otherwise such portion of data is discarded !
        Defaults to 5 timesteps.
    time_is_end_interval: bool
        Whether time refers to the end of the measurement interval.
        The default is True.

    Returns
    -------
    dict[int, xr.Dataset]
        A dictionary where keys are the identified sampling intervals (in seconds),
        and values are xarray.Datasets containing only data from those sampling intervals.
    """
    # Define array of possible measurement intervals
    measurement_intervals = np.array(measurement_intervals)

    # Check sorted by time and sort if necessary
    ds = ensure_sorted_by_time(ds)

    # If a single measurement interval expected, return dictionary with input dataset
    if len(measurement_intervals) == 1:
        dict_ds = {int(measurement_intervals[0]): ds}
        return dict_ds

    # If sample_interval is a dataset variable, use it to define dictionary of datasets
    if "sample_interval" in ds:
        return {int(interval): ds.isel(time=ds["sample_interval"] == interval) for interval in measurement_intervals}

    # ---------------------------------------------------------------------------------------.
    # Otherwise exploit difference between timesteps to identify change point

    # Calculate time differences in seconds
    deltadt = np.abs(np.diff(ds["time"].data)).astype("timedelta64[s]").astype(int)

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

    # Check which measurements intervals are occurring in the dataset
    uniques = np.unique(mapped_intervals)
    uniques_intervals = uniques[~np.isnan(uniques)]
    n_different_intervals_occurring = len(uniques_intervals)
    if n_different_intervals_occurring == 1:
        dict_ds = {int(k): ds for k in uniques_intervals}
        return dict_ds

    # Fill NaNs: decide whether to attach to previous or next interval
    for i in range(len(mapped_intervals)):
        if np.isnan(mapped_intervals[i]):
            # If next exists and is NaN → forward fill
            if i + 1 < len(mapped_intervals) and np.isnan(mapped_intervals[i + 1]):
                mapped_intervals[i] = mapped_intervals[i - 1] if i > 0 else mapped_intervals[i + 1]
            # Otherwise → backward fill (attach to next valid)
            else:
                mapped_intervals[i] = (
                    mapped_intervals[i + 1] if i + 1 < len(mapped_intervals) else mapped_intervals[i - 1]
                )

    # Infill np.nan values by using neighbor intervals
    # Forward fill
    # for i in range(1, len(mapped_intervals)):
    #     if np.isnan(mapped_intervals[i]):
    #         mapped_intervals[i] = mapped_intervals[i - 1]

    # # Backward fill (in case the first entries were np.nan)
    # for i in range(len(mapped_intervals) - 2, -1, -1):
    #     if np.isnan(mapped_intervals[i]):
    #         mapped_intervals[i] = mapped_intervals[i + 1]

    # Now all intervals are assigned to one of the possible measurement_intervals.
    # Identify boundaries where interval changes
    change_points = np.where(mapped_intervals[:-1] != mapped_intervals[1:])[0] + 1

    # Split ds into segments according to change_points
    offset = 1 if time_is_end_interval else 0
    segments = np.split(np.arange(ds.sizes["time"]), change_points + offset)

    # Remove segments with less than min_block_size elements
    segments = [seg for seg in segments if len(seg) >= min_block_size]
    if len(segments) == 0:
        raise ValueError(
            f"No blocks of {min_block_size} consecutive timesteps with constant sampling interval are available.",
        )

    # Define dataset indices for each sampling interva
    dict_sampling_interval_indices = {}
    used_indices = set()
    for seg in segments:
        # Define the assumed sampling interval of such segment
        start_idx = seg[0]
        segment_sampling_interval = int(mapped_intervals[start_idx])
        # Remove any indices that have already been assigned to another interval
        seg_filtered = seg[~np.isin(seg, list(used_indices))]

        # Only keep segment if it still meets minimum size after filtering
        if len(seg_filtered) >= min_block_size:
            if segment_sampling_interval not in dict_sampling_interval_indices:
                dict_sampling_interval_indices[segment_sampling_interval] = [seg_filtered]
            else:
                dict_sampling_interval_indices[segment_sampling_interval].append(seg_filtered)

            # Mark these indices as used
            used_indices.update(seg_filtered)

    # Concatenate indices for each sampling interval
    dict_sampling_interval_indices = {
        k: np.concatenate(list_indices)
        for k, list_indices in dict_sampling_interval_indices.items()
        if list_indices  # Only include if there are valid segments
    }

    # Define dictionary of datasets
    dict_ds = {int(k): ds.isel(time=indices) for k, indices in dict_sampling_interval_indices.items()}
    return dict_ds


####---------------------------------------------------------------------------------
#### Timesteps duplicates


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


####---------------------------------------------------------------------------------
#### Timesteps regularization


def get_problematic_timestep_indices(timesteps, sample_interval):
    """Identify timesteps with missing previous or following timesteps."""
    previous_time = timesteps - pd.Timedelta(seconds=sample_interval)
    next_time = timesteps + pd.Timedelta(seconds=sample_interval)
    idx_previous_missing = np.where(~np.isin(previous_time, timesteps))[0][1:]
    idx_next_missing = np.where(~np.isin(next_time, timesteps))[0][:-1]
    idx_isolated_missing = np.intersect1d(idx_previous_missing, idx_next_missing)
    idx_previous_missing = idx_previous_missing[np.isin(idx_previous_missing, idx_isolated_missing, invert=True)]
    idx_next_missing = idx_next_missing[np.isin(idx_next_missing, idx_isolated_missing, invert=True)]
    return idx_previous_missing, idx_next_missing, idx_isolated_missing


def regularize_timesteps(ds, sample_interval, robust=False, add_quality_flag=True, logger=None, verbose=True):
    """Ensure timesteps match with the sample_interval.

    This function:
    - drop dataset indices with duplicated timesteps,
    - but does not add missing timesteps to the dataset.
    """
    # Check sorted by time and sort if necessary
    ds = ensure_sorted_by_time(ds)

    # Convert time to pandas.DatetimeIndex for easier manipulation
    times = pd.to_datetime(ds["time"].to_numpy())

    # Determine the start and end times
    start_time = times[0].floor(f"{sample_interval}s")
    end_time = times[-1].ceil(f"{sample_interval}s")

    # Create the expected time grid
    expected_times = pd.date_range(start=start_time, end=end_time, freq=f"{sample_interval}s")

    # Convert to numpy arrays
    times = times.to_numpy(dtype="M8[s]")
    expected_times = expected_times.to_numpy(dtype="M8[s]")

    # Map original times to the nearest expected times
    # Calculate the difference between original times and expected times
    time_deltas = np.abs(times - expected_times[:, None]).astype(int)

    # Find the index of the closest expected time for each original time
    nearest_indices = np.argmin(time_deltas, axis=0)
    adjusted_times = expected_times[nearest_indices]

    # Check for duplicates in adjusted times
    unique_times, counts = np.unique(adjusted_times, return_counts=True)
    duplicates = unique_times[counts > 1]

    # Initialize time quality flag
    # - 0 when ok or just rounded to closest 00
    # - 1 if previous timestep is missing
    # - 2 if next timestep is missing
    # - 3 if previous and next timestep is missing
    # - 4 if solved duplicated timesteps
    # - 5 if needed to drop duplicated timesteps and select the last
    flag_previous_missing = 1
    flag_next_missing = 2
    flag_isolated_timestep = 3
    flag_solved_duplicated_timestep = 4
    flag_dropped_duplicated_timestep = 5
    qc_flag = np.zeros(adjusted_times.shape)

    # Initialize list with the duplicated timesteps index to drop
    # - We drop the first occurrence because is likely the shortest interval
    idx_to_drop = []

    # Attempt to resolve for duplicates
    if duplicates.size > 0:
        # Handle duplicates
        for dup_time in duplicates:
            # Indices of duplicates
            dup_indices = np.where(adjusted_times == dup_time)[0]
            n_duplicates = len(dup_indices)
            # Define previous and following timestep
            prev_time = dup_time - pd.Timedelta(seconds=sample_interval)
            next_time = dup_time + pd.Timedelta(seconds=sample_interval)
            # Try to find missing slots before and after
            # - If more than 3 duplicates, impossible to solve !
            count_solved = 0
            # If the previous timestep is available, set that one
            if n_duplicates == 2:
                if prev_time not in adjusted_times:
                    adjusted_times[dup_indices[0]] = prev_time
                    qc_flag[dup_indices[0]] = flag_solved_duplicated_timestep
                    count_solved += 1
                elif next_time not in adjusted_times:
                    adjusted_times[dup_indices[-1]] = next_time
                    qc_flag[dup_indices[-1]] = flag_solved_duplicated_timestep
                    count_solved += 1
                else:
                    pass
            elif n_duplicates == 3:
                if prev_time not in adjusted_times:
                    adjusted_times[dup_indices[0]] = prev_time
                    qc_flag[dup_indices[0]] = flag_solved_duplicated_timestep
                    count_solved += 1
                if next_time not in adjusted_times:
                    adjusted_times[dup_indices[-1]] = next_time
                    qc_flag[dup_indices[-1]] = flag_solved_duplicated_timestep
                    count_solved += 1
            if count_solved != n_duplicates - 1:
                idx_to_drop = np.append(idx_to_drop, dup_indices[0:-1])
                qc_flag[dup_indices[-1]] = flag_dropped_duplicated_timestep
                msg = (
                    f"Cannot resolve {n_duplicates} duplicated timesteps "
                    f"(after trailing seconds correction) around {dup_time}."
                )
                log_warning(logger=logger, msg=msg, verbose=verbose)
                if robust:
                    raise ValueError(msg)

    # Update the time coordinate (Convert to ns for xarray compatibility)
    ds = ds.assign_coords({"time": adjusted_times.astype("datetime64[ns]")})

    # Update quality flag values for next and previous timestep is missing
    if add_quality_flag:
        idx_previous_missing, idx_next_missing, idx_isolated_missing = get_problematic_timestep_indices(
            adjusted_times,
            sample_interval,
        )
        qc_flag[idx_previous_missing] = np.maximum(qc_flag[idx_previous_missing], flag_previous_missing)
        qc_flag[idx_next_missing] = np.maximum(qc_flag[idx_next_missing], flag_next_missing)
        qc_flag[idx_isolated_missing] = np.maximum(qc_flag[idx_isolated_missing], flag_isolated_timestep)

        # If the first timestep is at 00:00 and currently flagged as previous missing (1), reset to 0
        # first_time = pd.to_datetime(adjusted_times[0]).time()
        # first_expected_time = pd.Timestamp("00:00:00").time()
        # if first_time == first_expected_time and qc_flag[0] == flag_previous_missing:
        #     qc_flag[0] = 0

        # # If the last timestep is flagged and currently flagged as next missing (2), reset it to 0
        # last_time = pd.to_datetime(adjusted_times[-1]).time()
        # last_time_expected = (pd.Timestamp("00:00:00") - pd.Timedelta(30, unit="seconds")).time()
        # # Check if adding one interval would go beyond the end_time
        # if last_time == last_time_expected and qc_flag[-1] == flag_next_missing:
        #     qc_flag[-1] = 0

        # Assign time quality flag coordinate
        ds["time_qc"] = xr.DataArray(qc_flag, dims="time")
        ds = ds.set_coords("time_qc")

        # Add CF attributes for time_qc
        ds["time_qc"].attrs = {
            "long_name": "time quality flag",
            "standard_name": "status_flag",
            "units": "1",
            "valid_range": [0, 5],
            "flag_values": [0, 1, 2, 3, 4, 5],
            "flag_meanings": (
                "good_data "
                "previous_timestep_missing "
                "next_timestep_missing "
                "isolated_timestep "
                "solved_duplicated_timestep "
                "dropped_duplicated_timestep"
            ),
            "comment": (
                "Quality flag for time coordinate. "
                "Flag 0: data is good or just rounded to nearest sampling interval. "
                "Flag 1: previous timestep is missing in the time series. "
                "Flag 2: next timestep is missing in the time series. "
                "Flag 3: both previous and next timesteps are missing (isolated timestep). "
                "Flag 4: timestep was moved from duplicate to fill missing timestep. "
                "Flag 5: duplicate timestep was dropped, keeping the last occurrence."
            ),
        }

    # Drop duplicated timesteps
    # - Using ds =  ds.drop_isel({"time": idx_to_drop.astype(int)}) raise:
    #   --> pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects
    #   --> https://github.com/pydata/xarray/issues/6605
    if len(idx_to_drop) > 0:
        idx_to_drop = idx_to_drop.astype(int)
        idx_valid_timesteps = np.arange(0, ds["time"].size)
        idx_valid_timesteps = np.delete(idx_valid_timesteps, idx_to_drop)
        ds = ds.isel(time=idx_valid_timesteps)
    # Return dataset
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
    mask = unique_deltadt == sample_interval
    sample_interval_counts = counts[mask].item() if mask.any() else 0
    sample_interval_fraction = fractions[mask].item() if mask.any() else 0.0

    # Compute stats about most frequent deltadt
    mask = unique_deltadt == most_frequent_deltadt
    most_frequent_deltadt_counts = counts[mask].item() if mask.any() else 0
    most_frequent_deltadt_fraction = fractions[mask].item() if mask.any() else 0.0

    # Compute stats about unexpected deltadt
    unexpected_intervals = unique_deltadt[unique_deltadt != sample_interval]
    unexpected_intervals_counts = counts[unique_deltadt != sample_interval]
    unexpected_intervals_fractions = fractions[unique_deltadt != sample_interval]
    frequent_unexpected_intervals = unexpected_intervals[unexpected_intervals_fractions > 5]

    # Report warning if the sampling_interval deltadt occurs less often than 60 % of times
    # -> TODO: maybe only report in stations where the disdro does not log only data when rainy
    if sample_interval_fraction < 60:
        msg = (
            f"The expected (sampling) interval between observations occurs only "
            f"{sample_interval_counts}/{n} times ({sample_interval_fraction} %)."
        )
        log_warning(logger=logger, msg=msg, verbose=verbose)

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
        msg = "The following time intervals between observations occur frequently: "
        for interval in frequent_unexpected_intervals:
            c = unexpected_intervals_counts[unexpected_intervals == interval].item()
            f = unexpected_intervals_fractions[unexpected_intervals == interval].item()
            msg = msg + f"{interval} s ({f}%) ({c}/{n})"
        log_warning(logger=logger, msg=msg, verbose=verbose)
    return ds


####----------------------------------------------------------------------------------------------.
#### Wrapper


def _finalize_l0c_dataset(ds, sample_interval, sensor_name, verbose=True, logger=None):
    """Finalize a L0C dataset with unique sampling interval.

    It adds the sampling_interval coordinate and it regularizes the timesteps for trailing seconds.
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
    # - Do not discard anything
    # - Just log warnings in the log file
    ds = check_timesteps_regularity(ds=ds, sample_interval=sample_interval, verbose=verbose, logger=logger)

    # Shift timesteps to ensure time correspond to start of measurement interval
    # TODO as function of sensor name

    # Set netCDF dimension order
    # --> Required for correct encoding !
    ds = ds.transpose("time", "diameter_bin_center", ...)

    # Set encodings
    ds = set_l0b_encodings(ds=ds, sensor_name=sensor_name)

    # Update global attributes
    ds = set_disdrodb_attrs(ds, product="L0C")
    return ds


def create_l0c_datasets(
    event_info,
    measurement_intervals,
    sensor_name,
    ensure_variables_equality=True,
    logger=None,
    verbose=True,
):
    """
    Create a single dataset by merging and processing data from multiple filepaths.

    Parameters
    ----------
    event_info : dict
        Dictionary with start_time, end_time and filepaths keys.

    Returns
    -------
    dict
        A dictionary with an xarray.Dataset for each measurement interval.

    Raises
    ------
    ValueError
        If less than 5 timesteps are available for the specified day.

    Notes
    -----
    - Data is loaded into memory and connections to source files are closed before returning the dataset.
    - Tolerance in input files is used around expected dataset start_time and end_time to account for
      imprecise logging times and ensuring correct definition of qc_time at files boundaries (e.g. 00:00).
    - Duplicated timesteps with different raw drop number values are dropped
    - First occurrence of duplicated timesteps with equal raw drop number values is kept.
    - Regularizes timesteps to handle trailing seconds.
    """
    # ---------------------------------------------------------------------------------------.
    # Retrieve information
    start_time = np.array(event_info["start_time"], dtype="M8[s]")
    end_time = np.array(event_info["end_time"], dtype="M8[s]")
    filepaths = event_info["filepaths"]

    # Define expected dataset time coverage
    start_time_tol = start_time - np.array(TOLERANCE_SECONDS, dtype="m8[s]")
    end_time_tol = end_time + np.array(TOLERANCE_SECONDS, dtype="m8[s]")

    # ---------------------------------------------------------------------------------------.
    # Open files with data within the provided day and concatenate them
    ds = open_netcdf_files(
        filepaths,
        start_time=start_time_tol,
        end_time=end_time_tol,
        chunks={},
        parallel=False,
        compute=True,
    )
    
    # If not data for that time block, return empty dictionary
    # - Can occur when raw files are already by block of months and e.g. here saving to daily blocks !
    if ds.sizes["time"] == 0:
        log_info(logger=logger, msg=f"No data between {start_time} and {end_time}.", verbose=verbose)
        return {}

    # ---------------------------------------------------------------------------------------.
    # If sample interval is a dataset variable, drop timesteps with unexpected measurement intervals !
    if "sample_interval" in ds:
        ds = drop_timesteps_with_invalid_sample_interval(
            ds=ds,
            measurement_intervals=measurement_intervals,
            verbose=verbose,
            logger=logger,
        )
        n_timesteps = len(ds["time"])
        if n_timesteps < 3:
            raise ValueError(f"Only {n_timesteps} timesteps left after removing those with unexpected sample interval.")

    # ---------------------------------------------------------------------------------------.
    # Remove duplicated timesteps (before correcting for trailing seconds)
    # - It checks that duplicated timesteps have the same raw_drop_number values
    # - If duplicated timesteps have different raw_drop_number values:
    #   --> warning is raised
    #   --> timesteps are dropped
    ds = remove_duplicated_timesteps(
        ds,
        ensure_variables_equality=ensure_variables_equality,
        logger=logger,
        verbose=verbose,
    )

    # Raise error if less than 3 timesteps left
    n_timesteps = len(ds["time"])
    if n_timesteps < 3:
        raise ValueError(f"{n_timesteps} timesteps left after removing duplicated.")

    # ---------------------------------------------------------------------------------------.
    # Split dataset by sampling intervals
    dict_ds = split_dataset_by_sampling_intervals(
        ds=ds,
        measurement_intervals=measurement_intervals,
        min_sample_interval=10,
        min_block_size=5,
    )

    # Log a warning if two sampling intervals are present within a given time block
    if len(dict_ds) > 1:
        occuring_sampling_intervals = list(dict_ds)
        msg = f"The input files contains these sampling intervals: {occuring_sampling_intervals}."
        log_warning(logger=logger, msg=msg, verbose=verbose)

    # ---------------------------------------------------------------------------------------.
    # Finalize L0C datasets
    # - Add and ensure sample_interval coordinate has just 1 value (not varying with time)
    # - Regularize timesteps for trailing seconds
    dict_ds = {
        sample_interval: _finalize_l0c_dataset(
            ds=ds,
            sample_interval=sample_interval,
            sensor_name=sensor_name,
            verbose=verbose,
            logger=logger,
        ).sel({"time": slice(start_time, end_time)})
        for sample_interval, ds in dict_ds.items()
    }
    return dict_ds
