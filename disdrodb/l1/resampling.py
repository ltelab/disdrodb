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
"""Utilities for temporal resampling."""
import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.utils.time import ensure_sample_interval_in_seconds, regularize_dataset

DEFAULT_ACCUMULATIONS = ["10s", "30s", "1min", "2min", "5min", "10min", "30min", "1hour"]


def add_sample_interval(ds, sample_interval):
    """Add a sample_interval coordinate to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to which the sample_interval coordinate will be added.
    sample_interval : int or float
        The dataset sample interval in seconds.

    Returns
    -------
    xarray.Dataset
        The dataset with the added sample interval coordinate.

    Notes
    -----
    The function adds a new coordinate named 'sample_interval' to the dataset and
    updates the 'measurement_interval' attribute.
    """
    # Add sample_interval coordinate
    ds["sample_interval"] = sample_interval
    ds["sample_interval"].attrs["description"] = "Sample interval"
    ds["sample_interval"].attrs["long_name"] = "Sample interval"
    ds["sample_interval"].attrs["units"] = "seconds"
    ds = ds.set_coords("sample_interval")
    # Update measurement_interval attribute
    ds.attrs = ds.attrs.copy()
    ds.attrs["measurement_interval"] = int(sample_interval)
    return ds


def define_window_size(sample_interval, accumulation_interval):
    """
    Calculate the rolling window size based on sampling and accumulation intervals.

    Parameters
    ----------
    sampling_interval : int
        The sampling interval in seconds.
    accumulation_interval : int
        The desired accumulation interval in seconds.

    Returns
    -------
    int
        The calculated window size as the number of sampling intervals required to cover the accumulation interval.

    Raises
    ------
    ValueError
        If the accumulation interval is not a multiple of the sampling interval.

    Examples
    --------
    >>> define_window_size(60, 300)
    5

    >>> define_window_size(120, 600)
    5
    """
    # Check compatitiblity
    if accumulation_interval % sample_interval != 0:
        raise ValueError("The accumulation interval must be a multiple of the sample interval.")

    # Calculate the window size
    window_size = accumulation_interval // sample_interval

    return window_size


def _resample(ds, variables, accumulation, op):
    if not variables:
        return {}
    ds_subset = ds[variables]
    if "time" in ds_subset.dims:
        return getattr(ds_subset.resample({"time": accumulation}), op)(skipna=False)
    return ds_subset


def _rolling(ds, variables, window_size, op):
    if not variables:
        return {}
    ds_subset = ds[variables]
    if "time" in ds_subset.dims:
        return getattr(ds_subset.rolling(time=window_size, center=False), op)(skipna=False)
    return ds_subset


def resample_dataset(ds, sample_interval, accumulation_interval, rolling=True):
    """
    Resample the dataset to a specified accumulation interval.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be resampled.
    sample_interval : int
        The sample interval of the input dataset.
    accumulation_interval : int
        The interval in seconds over which to accumulate the data.
    rolling : bool, optional
        If True, apply a rolling window before resampling. Default is True.
        If True, forward rolling is performed.
        The output timesteps correspond to the starts of the periods over which
        the resampling operation has been performed !

    Returns
    -------
    xarray.Dataset
        The resampled dataset with updated attributes.

    Notes
    -----
    - The function regularizes the dataset (infill possible missing timesteps)
      before performing the resampling operation.
    - Variables are categorized into those to be averaged, accumulated, minimized, and maximized.
    - Custom processing for quality flags and handling of NaNs is defined.
    - The function updates the dataset attributes and the sample_interval coordinate.

    """
    # --------------------------------------------------------------------------.
    # Ensure sample interval in seconds
    sample_interval = int(ensure_sample_interval_in_seconds(sample_interval))

    # --------------------------------------------------------------------------.
    # Raise error if the accumulation_interval is less than the sample interval
    if accumulation_interval < sample_interval:
        raise ValueError("Expecting an accumulation_interval > sample interval.")
    # Raise error if accumulation_interval is not multiple of sample_interval
    if not accumulation_interval % sample_interval == 0:
        raise ValueError("The accumulation_interval is not a multiple of sample interval.")

    # --------------------------------------------------------------------------.
    #### Preprocess the dataset
    # Here we set NaN in the raw_drop_number to 0
    # - We assume that NaN corresponds to 0
    # - When we regularize, we infill with NaN
    # - When we aggregate with sum, we don't skip NaN
    # --> Aggregation with original missing timesteps currently results in NaN !

    # Infill NaN values with zeros for drop_number and raw_drop_number
    # - This might alter integrated statistics if NaN in spectrum does not actually correspond to 0 !
    # - TODO: NaN should not be set as 0 !
    for var in ["drop_number", "raw_drop_number"]:
        if var in ds:
            ds[var] = xr.where(np.isnan(ds[var]), 0, ds[var])

    # Ensure regular dataset without missing timesteps
    # --> This adds NaN values for missing timesteps
    ds = regularize_dataset(ds, freq=f"{sample_interval}s")

    # --------------------------------------------------------------------------.
    # Define dataset attributes
    attrs = ds.attrs.copy()
    if rolling:
        attrs["disdrodb_rolled_product"] = "True"
    else:
        attrs["disdrodb_rolled_product"] = "False"

    if sample_interval == accumulation_interval:
        attrs["disdrodb_aggregated_product"] = "False"
        ds = add_sample_interval(ds, sample_interval=accumulation_interval)
        ds.attrs = attrs
        return ds

    # --------------------------------------------------------------------------.
    # Resample the dataset
    attrs["disdrodb_aggregated_product"] = "True"

    # Initialize resample dataset
    ds_resampled = xr.Dataset()

    # Retrieve variables to average/sum
    var_to_average = ["fall_velocity"]
    var_to_cumulate = ["raw_drop_number", "drop_number", "drop_counts", "N", "Nraw", "Nremoved"]
    var_to_min = ["Dmin"]
    var_to_max = ["Dmax"]

    # Retrieve available variables
    var_to_average = [var for var in var_to_average if var in ds]
    var_to_cumulate = [var for var in var_to_cumulate if var in ds]
    var_to_min = [var for var in var_to_min if var in ds]
    var_to_max = [var for var in var_to_max if var in ds]

    # TODO Define custom processing
    # - quality_flag --> take worst
    # - skipna if less than fraction (to not waste lot of data when aggregating over i.e. hours)
    # - Add tolerance on fraction of missing timesteps for large accumulation_intervals

    # Resample the dataset
    # - Rolling currently does not allow direct rolling forward.
    # - We currently use center=False which means search for data backward (right-aligned) !
    # - We then drop the first 'window_size' NaN timesteps and we shift backward the timesteps.
    # - https://github.com/pydata/xarray/issues/9773
    # - https://github.com/pydata/xarray/issues/8958
    if not rolling:
        # Resample
        accumulation = pd.Timedelta(seconds=accumulation_interval)
        ds_resampled.update(_resample(ds=ds, variables=var_to_average, accumulation=accumulation, op="mean"))
        ds_resampled.update(_resample(ds=ds, variables=var_to_cumulate, accumulation=accumulation, op="sum"))
        ds_resampled.update(_resample(ds=ds, variables=var_to_min, accumulation=accumulation, op="min"))
        ds_resampled.update(_resample(ds=ds, variables=var_to_max, accumulation=accumulation, op="max"))
    else:
        # Roll and Resample
        window_size = define_window_size(sample_interval=sample_interval, accumulation_interval=accumulation_interval)
        ds_resampled.update(_rolling(ds=ds, variables=var_to_average, window_size=window_size, op="mean"))
        ds_resampled.update(_rolling(ds=ds, variables=var_to_cumulate, window_size=window_size, op="sum"))
        ds_resampled.update(_rolling(ds=ds, variables=var_to_min, window_size=window_size, op="min"))
        ds_resampled.update(_rolling(ds=ds, variables=var_to_max, window_size=window_size, op="max"))
        # Ensure time to correspond to the start time of the measurement period
        ds_resampled = ds_resampled.isel(time=slice(window_size - 1, None)).assign_coords(
            {"time": ds_resampled["time"].data[: -window_size + 1]},
        )

    # Add attributes
    ds_resampled.attrs = attrs

    # Add accumulation_interval as new sample_interval coordinate
    ds_resampled = add_sample_interval(ds_resampled, sample_interval=accumulation_interval)
    return ds_resampled
