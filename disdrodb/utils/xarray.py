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
"""Xarray utility."""

import pandas as pd
import xarray as xr
from xarray.core import dtypes


def get_dataset_start_end_time(ds: xr.Dataset):
    """Retrieves dataset starting and ending time.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    tuple
        (starting_time, ending_time)

    """
    starting_time = ds["time"].values[0]
    ending_time = ds["time"].values[-1]
    return (starting_time, ending_time)


def regularize_dataset(ds: xr.Dataset, freq: str, time_dim="time", method=None, fill_value=dtypes.NA):
    """
    Regularize a dataset across time dimension with uniform resolution.

    Parameters
    ----------
    ds : xr.Dataset
        DESCRIPTION.
    time_dim : TYPE, optional
        DESCRIPTION. The default is "time".
    freq : str
        The `freq` string to pass to pd.date_range to define the new time coordinates.
        Examples: freq="2min"
    time_dim : TYPE, optional
        The time dimension in the xr.Dataset. The default is "time".
    method : TYPE, optional
        Method to use for filling missing timesteps.
        If None, fill with fill_value. The default is None.
        For other possible methods, see https://docs.xarray.dev/en/stable/generated/xarray.Dataset.reindex.html
    fill_value : float, optional
        Fill value to fill missing timesteps. The default is dtypes.NA.

    Returns
    -------
    ds_reindexed : xr.Dataset
        Regularized dataset.

    """
    start = ds[time_dim].values[0]
    end = ds[time_dim].values[-1]
    new_time_index = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=freq)

    # Regularize dataset and fill with NA values
    ds_reindexed = ds.reindex(
        {"time": new_time_index},
        method=method,  # do not fill gaps
        # tolerance=tolerance,  # mismatch in seconds
        fill_value=fill_value,
    )
    return ds_reindexed
