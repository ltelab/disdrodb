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

from disdrodb.utils.time import (
    ensure_sample_interval_in_seconds,
    get_dataset_start_end_time,
    get_sampling_information,
    regularize_dataset,
)


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


def _finalize_qc_resampling(ds, sample_interval, accumulation_interval):
    # Compute qc_resampling
    # - 0 if not missing timesteps
    # - 1 if all timesteps missing
    n_timesteps = accumulation_interval / sample_interval
    ds["qc_resampling"] = np.round(1 - ds["qc_resampling"] / n_timesteps, 1)
    ds["qc_resampling"].attrs = {
        "long_name": "Resampling Quality Control Flag",
        "standard_name": "quality_flag",
        "units": "",
        "valid_min": 0.0,
        "valid_max": 1.0,
        "description": (
            "Fraction of timesteps missing when resampling the data."
            "0 = No timesteps missing; 1 = All timesteps missing;"
            "Intermediate values indicate partial data coverage."
        ),
    }
    return ds


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


def resample_dataset(ds, sample_interval, temporal_resolution):
    """
    Resample the dataset to a specified accumulation interval.

    The output timesteps correspond to the starts of the periods over which
    the resampling operation has been performed !

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be resampled.
    sample_interval : int
        The sample interval (in seconds) of the input dataset.
    temporal_resolution : str
        The desired temporal resolution for resampling.
        It should be a string representing the accumulation interval,
        e.g., "5MIN" for 5 minutes, "1H" for 1 hour, "30S" for 30 seconds, etc.
        Prefixed with "ROLL" for rolling resampling, e.g., "ROLL5MIN".

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
    from disdrodb.l1.classification import TEMPERATURE_VARIABLES

    # --------------------------------------------------------------------------.
    # Ensure sample interval in seconds
    sample_interval = int(ensure_sample_interval_in_seconds(sample_interval))

    # Retrieve accumulation_interval and rolling option
    accumulation_interval, rolling = get_sampling_information(temporal_resolution)

    # --------------------------------------------------------------------------.
    # Raise error if the accumulation_interval is less than the sample interval
    if accumulation_interval < sample_interval:
        raise ValueError("Expecting an accumulation_interval > sample interval.")
    # Raise error if accumulation_interval is not multiple of sample_interval
    if not accumulation_interval % sample_interval == 0:
        raise ValueError("The accumulation_interval is not a multiple of sample interval.")

    # Retrieve input dataset start_time and end_time
    start_time, end_time = get_dataset_start_end_time(ds, time_dim="time")

    # Initialize qc_resampling
    ds["qc_resampling"] = xr.ones_like(ds["time"], dtype="float")

    # Retrieve dataset attributes
    attrs = ds.attrs.copy()

    # If no resampling, return as it is
    if sample_interval == accumulation_interval:
        attrs["disdrodb_aggregated_product"] = "False"
        attrs["disdrodb_rolled_product"] = "False"
        attrs["disdrodb_temporal_resolution"] = temporal_resolution

        ds = _finalize_qc_resampling(ds, sample_interval=sample_interval, accumulation_interval=accumulation_interval)
        ds = add_sample_interval(ds, sample_interval=accumulation_interval)
        ds.attrs = attrs
        return ds

    # --------------------------------------------------------------------------.
    #### Preprocess the dataset
    # - Set timesteps with NaN in drop_number to zero (and set qc_resampling to 0)
    # - When we aggregate with sum, we don't skip NaN
    #   --> Resampling over missing timesteps will result in NaN drop_number and qc_resampling = 1
    #   --> Resampling over timesteps with NaN in drop_number will result in finite drop_number but qc_resampling > 0
    # - qc_resampling will inform on the amount of timesteps missing

    for var in ["drop_number", "raw_drop_number", "drop_counts", "drop_number_concentration"]:
        if var in ds:
            dims = set(ds[var].dims) - {"time"}
            invalid_timesteps = np.isnan(ds[var]).any(dim=dims)
            ds[var] = ds[var].where(~invalid_timesteps, 0)
            ds["qc_resampling"] = ds["qc_resampling"].where(~invalid_timesteps, 0)

            if np.all(invalid_timesteps).item():
                raise ValueError("No timesteps with valid spectrum.")

    # Ensure regular dataset without missing timesteps
    # --> This adds NaN values for missing timesteps
    ds = regularize_dataset(ds, freq=f"{sample_interval}s", start_time=start_time, end_time=end_time)
    ds["qc_resampling"] = ds["qc_resampling"].where(~np.isnan(ds["qc_resampling"]), 0)

    # --------------------------------------------------------------------------.
    # Define dataset attributes
    if rolling:
        attrs["disdrodb_rolled_product"] = "True"
    else:
        attrs["disdrodb_rolled_product"] = "False"

    attrs["disdrodb_aggregated_product"] = "True"
    attrs["disdrodb_temporal_resolution"] = temporal_resolution

    # --------------------------------------------------------------------------.
    # Initialize resample dataset
    ds_resampled = xr.Dataset()

    # Retrieve variables to average/sum
    # - ATTENTION: it will not resample non-dimensional time coordinates of the dataset !
    # - precip_flag used for OceanRain ODM470 data
    var_to_average = ["fall_velocity"]
    var_to_cumulate = [
        "raw_drop_number",
        "drop_number",
        "drop_counts",
        "drop_number_concentration",
        "N",
        "Nraw",
        "Nremoved",
        "qc_resampling",
    ]
    var_to_min = ["Dmin", *TEMPERATURE_VARIABLES]
    var_to_max = ["Dmax", "qc_time", "precip_flag"]

    # Retrieve available variables
    var_to_average = [var for var in var_to_average if var in ds]
    var_to_cumulate = [var for var in var_to_cumulate if var in ds]
    var_to_min = [var for var in var_to_min if var in ds]
    var_to_max = [var for var in var_to_max if var in ds]

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

    # Finalize qc_resampling
    ds_resampled = _finalize_qc_resampling(
        ds_resampled,
        sample_interval=sample_interval,
        accumulation_interval=accumulation_interval,
    )
    # Set to NaN timesteps where qc_resampling == 1
    # --> This occurs for missing timesteps in input dataset or all NaN drop_number arrays
    variables = list(set(ds_resampled.data_vars) - {"qc_resampling"})
    mask_missing_timesteps = ds_resampled["qc_resampling"] != 1
    for var in variables:
        ds_resampled[var] = ds_resampled[var].where(mask_missing_timesteps)

    # Add attributes
    ds_resampled.attrs = attrs

    # Add accumulation_interval as new sample_interval coordinate
    ds_resampled = add_sample_interval(ds_resampled, sample_interval=accumulation_interval)
    return ds_resampled
