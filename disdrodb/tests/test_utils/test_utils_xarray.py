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

from disdrodb.utils.xarray import (
    regularize_dataset,
    xr_get_last_valid_idx,
)


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
    assert np.array_equal(ds_regularized["time"].to_numpy(), expected_times)

    # Get time index which were infilled
    new_indices = np.where(np.isin(expected_times, ds["time"].to_numpy(), invert=True))[0]
    assert np.all(ds_regularized.isel(time=new_indices)["data"].data == fill_value)


class TestXrGetLastValidIdx:
    """Test suite for the `xr_get_last_valid_idx` function."""

    def test_single_dimension_single_true(self):
        """Check correct last True index in a single-dimension array with one True."""
        da = xr.DataArray([False, False, True, False], dims=["x"])
        last_idx = xr_get_last_valid_idx(da, dim="x")
        # Default fill_value is dim_size - 1 (here 3), but we have a True at index 2
        expected = xr.DataArray(2)
        xr.testing.assert_equal(last_idx, expected)

    def test_single_dimension_multiple_true(self):
        """Check correct last True index in a single-dimension array with multiple Trues."""
        da = xr.DataArray([False, True, True, False], dims=["x"])
        last_idx = xr_get_last_valid_idx(da, dim="x")
        # Last True is at index 2
        expected = xr.DataArray(2)
        xr.testing.assert_equal(last_idx, expected)

    def test_single_dimension_all_false_default_fill_value(self):
        """Check default fill_value behavior when all values are False."""
        da = xr.DataArray([False, False, False], dims=["x"])
        # Default fill_value = dim_size - 1 = 2
        last_idx = xr_get_last_valid_idx(da, dim="x")
        expected = xr.DataArray(2)
        xr.testing.assert_equal(last_idx, expected)

    def test_single_dimension_all_false_nan_fill_value(self):
        """Check NaN fill_value behavior when all values are False."""
        da = xr.DataArray([False, False, False], dims=["x"])
        last_idx = xr_get_last_valid_idx(da, dim="x", fill_value=np.nan)
        # All false => we expect NaN
        expected = xr.DataArray(np.nan)
        xr.testing.assert_equal(last_idx, expected)

    def test_two_dimensions_mixed_true(self):
        """Check correct last True indices across an extra dimension."""
        da = xr.DataArray(
            [[False, True, True], [False, False, True], [True, False, False]],
            dims=["time", "feature"],
        )
        # We want the last True index along the "feature" dimension
        last_idx = xr_get_last_valid_idx(da, dim="feature")
        # For time=0: last True is index 2
        # For time=1: last True is index 2
        # For time=2: last True is index 0
        expected = xr.DataArray([2, 2, 0], dims=["time"])
        xr.testing.assert_equal(last_idx, expected)

    def test_two_dimensions_all_false(self):
        """Check behavior when an entire slice is False."""
        da = xr.DataArray(
            [[False, False], [False, True]],
            dims=["row", "col"],
        )
        # Last True index along 'col'
        last_idx = xr_get_last_valid_idx(da, dim="col")
        # For row=0: all false, default fill_value = 1 (because dim_size=2)
        # For row=1: last True is index 1
        expected = xr.DataArray([1, 1], dims=["row"])
        xr.testing.assert_equal(last_idx, expected)

    def test_two_dimensions_all_false_nan_fill_value(self):
        """Check NaN fill_value for slices that have no True values."""
        da = xr.DataArray(
            [[False, True], [False, False]],
            dims=["row", "col"],
        )
        last_idx = xr_get_last_valid_idx(da, dim="col", fill_value=np.nan)
        # For row=0: last True is index 1
        # For row=1: no True => NaN
        expected = xr.DataArray([1, np.nan], dims=["row"])
        xr.testing.assert_equal(last_idx, expected)

    def test_raises_value_error_non_boolean(self):
        """Check that a ValueError is raised if the DataArray is not boolean."""
        da = xr.DataArray([0.0, 1.0, 2.0], dims=["x"])
        with pytest.raises(ValueError, match="must be a boolean DataArray"):
            xr_get_last_valid_idx(da, dim="x")
