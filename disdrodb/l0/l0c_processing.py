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
import itertools
from disdrodb.api.info import get_start_end_time_from_filepaths
from mydsd.l1.resampling import add_sample_interval
from mydsd.utils.time import infer_sample_interval, regularize_timesteps


logger = logging.getLogger(__name__)


def get_files_per_days(filepaths):
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
    list_days = np.asanyarray(pd.date_range(start=start_day, end=end_day, freq='D')).astype("M8[D]")

    # Expand dimension to match each day using broadcasting
    files_start_time = files_start_time.astype("M8[D]")[:, np.newaxis]  # shape (n_files, 1)
    files_end_time = files_end_time.astype("M8[D]")[:, np.newaxis]      # shape (n_files, 1)
    
    # Create an array of all days
    # - Expand dimension to match each day using broadcasting
    days = list_days[np.newaxis, :]  # shape (1, n_days)
     
    # Use broadcasting to create a boolean matrix indicating which files cover which days
    mask = (files_start_time <= days) & (files_end_time >= days)  # shape (n_files, n_days)
    
    # np.sum(mask, axis=0) 
    
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
    intersecting_timesteps = np.intersect1d(da1["time"], da2["time"])
    da1_isel = np.where(np.isin(da1["time"], intersecting_timesteps))[0]
    da2_isel = np.where(np.isin(da2["time"], intersecting_timesteps))[0]
    return da1_isel, da2_isel


def check_same_raw_drop_number_values(list_ds, filepaths):
    # Retrieve variable to compare 
    list_drop_number = [ds["raw_drop_number"].compute() for ds in list_ds]
    # Compare values 
    combos =  list(itertools.combinations(range(len(list_drop_number)), 2)) 
    for i, j in combos:
        da1 = list_drop_number[i]
        da2 = list_drop_number[j]
        da1_isel, da2_isel = find_isel_common_time(da1=da1, da2=da2)
        if not np.all(da1.isel(time=da1_isel).data == da2.isel(time=da2_isel).data):
            file1 = filepaths[i]
            file2 = filepaths[i]
            msg = f"Duplicated timesteps have different values between file {file1} and {file2}"
            raise ValueError(msg)


def create_daily_file(day, filepaths):
    # Define start day and end of day
    start_day = np.array(day).astype("M8[D]")
    end_day = start_day + np.array(1, dtype="m8[D]") - np.array(1, dtype="m8[s]") # avoid 00:00 of next day ! 
    
    # Add tolerance for searching timesteps before and after 00:00 to account for unprecise logging time 
    # - Example: timestep 23:58 that should be 00.00 goes into the next day ... 
    start_day_tol = start_day - np.array(60, dtype="m8[s]")
    end_day_tol = end_day + np.array(60, dtype="m8[s]")
    
    # Open files with data within the provided day 
    # list_ds = [xr.open_dataset(filepath, chunks={}).sel({"time": slice(start_day_tol, end_day_tol)}) for filepath in filepaths]
    list_ds = [xr.open_dataset(filepath, chunks={}).sortby("time") for filepath in filepaths]
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
    sample_interval = infer_sample_interval(ds, verbose=False)
    ds = add_sample_interval(ds, sample_interval=sample_interval)

    # Regularize timesteps (for trailing seconds)
    ds = regularize_timesteps(ds, sample_interval=sample_interval)
    
    # Slice for requested day 
    ds = ds.sel({"time": slice(start_day, end_day)})
        
    # Load data into memory 
    ds = ds.compute() 
    
    # Close connection to source files 
    _ = [ds.close() for ds in list_ds]
    ds.close()
    del list_ds
    
    return ds