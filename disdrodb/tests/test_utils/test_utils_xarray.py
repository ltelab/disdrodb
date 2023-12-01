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
"""Test Xarray utility."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.utils.xarray import get_dataset_start_end_time, regularize_dataset


def create_test_dataset():
    """Create a mock xarray.Dataset for testing."""
    times = pd.date_range("2023-01-01", periods=10, freq="D")
    data = np.random.rand(10, 2, 2)  # Random data for the sake of example
    ds = xr.Dataset({"my_data": (("time", "x", "y"), data)}, coords={"time": times})
    return ds


def test_get_dataset_start_end_time():
    ds = create_test_dataset()
    expected_start_time = ds["time"].values[0]
    expected_end_time = ds["time"].values[-1]

    start_time, end_time = get_dataset_start_end_time(ds)

    assert start_time == expected_start_time
    assert end_time == expected_end_time

    # Test raise if empty dataset
    empty_ds = xr.Dataset()
    with pytest.raises(KeyError):
        get_dataset_start_end_time(empty_ds)


def test_regularize_dataset():
    # Create a sample Dataset
    times = pd.date_range("2020-01-01", periods=4, freq="2min")
    data = np.random.rand(4)
    ds = xr.Dataset({"data": ("time", data)}, coords={"time": times})

    # Regularize the dataset
    desired_freq = "1min"
    fill_value = 0
    ds_regularized = regularize_dataset(ds, freq=desired_freq, fill_value=fill_value)

    # Check new time dimension coordinates
    expected_times = pd.date_range("2020-01-01", periods=7, freq=desired_freq)
    assert np.array_equal(ds_regularized["time"].values, expected_times)

    # Get time index which were infilled
    new_indices = np.where(np.isin(expected_times, ds["time"].values, invert=True))[0]
    assert np.all(ds_regularized.isel(time=new_indices)["data"].data == fill_value)
