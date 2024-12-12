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
import itertools
import logging
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.api.info import get_start_end_time_from_filepaths
from disdrodb.l1.resampling import add_sample_interval
from disdrodb.utils.time import infer_sample_interval, regularize_timesteps

logger = logging.getLogger(__name__)


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
    # - Example: timestep 23:58 that should be 00.00 goes into the next day ...
    files_start_time = files_start_time - np.array(60, dtype="m8[s]")
    files_end_time = files_end_time + np.array(60, dtype="m8[s]")

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


def find_isel_common_time(da1, da2):
    """
    Find the indices of common time steps between two data arrays.

    Parameters
    ----------
    da1 : xarray.DataArray
        The first data array with a time coordinate.
    da2 : xarray.DataArray
        The second data array with a time coordinate.

    Returns
    -------
    da1_isel : numpy.ndarray
        Indices of the common time steps in the first data array.
    da2_isel : numpy.ndarray
        Indices of the common time steps in the second data array.

    Notes
    -----
    This function assumes that both input data arrays have a "time" coordinate.
    The function finds the intersection of the time steps in both data arrays
    and returns the indices of these common time steps for each data array.
    """
    intersecting_timesteps = np.intersect1d(da1["time"], da2["time"])
    da1_isel = np.where(np.isin(da1["time"], intersecting_timesteps))[0]
    da2_isel = np.where(np.isin(da2["time"], intersecting_timesteps))[0]
    return da1_isel, da2_isel


def check_same_raw_drop_number_values(list_ds, filepaths):
    """
    Check if the 'raw_drop_number' values are the same across multiple datasets.

    This function compares the 'raw_drop_number' values of multiple datasets to ensure they are identical
    at common timesteps.

    If any discrepancies are found, a ValueError is raised indicating which files
    have differing values.

    Parameters
    ----------
    list_ds : list of xarray.Dataset
        A list of xarray Datasets to be compared.
    filepaths : list of str
        A list of file paths corresponding to the datasets in `list_ds`.

    Raises
    ------
    ValueError
        If 'raw_drop_number' values differ at any common timestep between any two datasets.
    """
    # Retrieve variable to compare
    list_drop_number = [ds["raw_drop_number"].compute() for ds in list_ds]
    # Compare values
    combos = list(itertools.combinations(range(len(list_drop_number)), 2))
    for i, j in combos:
        da1 = list_drop_number[i]
        da2 = list_drop_number[j]
        da1_isel, da2_isel = find_isel_common_time(da1=da1, da2=da2)
        if not np.all(da1.isel(time=da1_isel).data == da2.isel(time=da2_isel).data):
            file1 = filepaths[i]
            file2 = filepaths[i]
            msg = f"Duplicated timesteps have different values between file {file1} and {file2}"
            raise ValueError(msg)


def create_daily_file(day, filepaths, verbose=True):
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
    # Define start day and end of day
    start_day = np.array(day).astype("M8[D]")
    end_day = start_day + np.array(1, dtype="m8[D]") - np.array(1, dtype="m8[s]")  # avoid 00:00 of next day !

    # Add tolerance for searching timesteps before and after 00:00 to account for imprecise logging time
    # - Example: timestep 23:58 that should be 00.00 goes into the next day ...
    start_day_tol = start_day - np.array(60, dtype="m8[s]")
    end_day_tol = end_day + np.array(60, dtype="m8[s]")

    # Open files with data within the provided day
    # list_ds = [xr.open_dataset(filepath, chunks={}).sel({"time": slice(start_day_tol, end_day_tol)})
    # for filepath in filepaths]
    list_ds = [xr.open_dataset(filepath, chunks={}, cache=False).sortby("time") for filepath in filepaths]
    list_ds = [ds.sel({"time": slice(start_day_tol, end_day_tol)}) for ds in list_ds]

    if len(list_ds) > 1:
        # Check duplicated timesteps have same raw_drop_number
        check_same_raw_drop_number_values(list_ds=list_ds, filepaths=filepaths)
        # Merge dataset
        ds = xr.merge(list_ds, join="outer", compat="no_conflicts", combine_attrs="override")
    else:
        ds = list_ds[0]

    # Check at least 5 timesteps are available
    if len(ds["time"]) < 5:
        raise ValueError(f"Less than 5 timesteps available for day {day}.")

    # Identify time integration
    sample_interval = infer_sample_interval(ds, verbose=verbose, robust=False)
    ds = add_sample_interval(ds, sample_interval=sample_interval)

    # Regularize timesteps (for trailing seconds)
    ds = regularize_timesteps(ds, sample_interval=sample_interval, robust=False, add_quality_flag=True)

    # Slice for requested day
    ds = ds.sel({"time": slice(start_day, end_day)})

    # Load data into memory
    ds = ds.compute()

    # Close connection to source files
    _ = [ds.close() for ds in list_ds]
    ds.close()
    del list_ds

    return ds


def copy_l0b_to_l0c_directory(filepath):
    """Copy L0B file to L0C directory."""
    import netCDF4

    # Copy file
    l0c_filepath = filepath.replace("L0B", "L0C")
    _ = shutil.copy(filepath, l0c_filepath)

    # Edit DISDRODB product attribute
    with netCDF4.Dataset(l0c_filepath, mode="a") as nc_file:
        # Modify the global attribute
        nc_file.setncattr("disdrodb_product", "L0C")
