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
import pytest
import xarray as xr
from xarray.core import dtypes

from disdrodb.utils.xarray import (
    define_dataarray_fill_value,
    define_fill_value_dictionary,
    xr_get_last_valid_idx,
    # remove_diameter_coordinates,
    # remove_velocity_coordinates,
)


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


class TestDefineDataarrayFillValue:
    def test_float_dtype_returns_NA(self):
        """Return NA for floating dtype DataArray."""
        da = xr.DataArray(np.array([1.1, 2.2], dtype=float))
        result = define_dataarray_fill_value(da)
        assert result is dtypes.NA

    def test_integer_dtype_uses_attrs_fillvalue(self):
        """Use _FillValue from attrs for integer dtype DataArray."""
        da = xr.DataArray(np.array([1, 2, 3], dtype=np.int32))
        da.attrs["_FillValue"] = -999
        result = define_dataarray_fill_value(da)
        assert result == -999

    def test_integer_dtype_uses_encoding_fillvalue(self):
        """Use _FillValue from encoding when attrs has none for integer dtype."""
        da = xr.DataArray(np.array([4, 5, 6], dtype=np.int16))
        # simulate NetCDF-style encoding
        da.encoding["_FillValue"] = -888
        result = define_dataarray_fill_value(da)
        assert result == -888

    def test_integer_dtype_without_fillvalue_returns_iinfo_max(self):
        """Return max integer value when no _FillValue is set for integer dtype."""
        da = xr.DataArray(np.array([7, 8, 9], dtype=np.int8))
        expected = np.iinfo(np.int8).max
        result = define_dataarray_fill_value(da)
        assert result == expected

    def test_non_numeric_dtype_returns_None(self):
        """Return None for non-numeric dtype DataArray."""
        da = xr.DataArray(np.array(["a", "b", "c"], dtype=object))
        result = define_dataarray_fill_value(da)
        assert result is None


class TestDefineFillValueDictionary:

    def test_dataarray_float_dtype(self):
        """Return dict with NA for floating DataArray."""
        da = xr.DataArray(
            np.array([1.0, 2.0], dtype=float),
            dims="x",
            coords={"x": np.array([0, 1], dtype=int)},
            name="var",
        )
        result = define_fill_value_dictionary(da)
        # Should include only the DataArray and its coordinate
        assert set(result.keys()) == {"var", "x"}
        assert result["var"] is dtypes.NA
        assert result["x"] == np.iinfo(np.int64).max

    def test_dataarray_int_dtype_with_attrs(self):
        """Return dict with attrs fill value for integer DataArray."""
        da = xr.DataArray(
            np.array([1, 2, 3], dtype=np.int32),
            dims="y",
            coords={"y": np.array([10, 20, 30], dtype=int)},
            name="var2",
        )
        da.attrs["_FillValue"] = -1
        result = define_fill_value_dictionary(da)
        assert set(result.keys()) == {"var2", "y"}
        assert result["var2"] == -1
        assert result["y"] == np.iinfo(np.int64).max

    def test_dataset_mixed_dtypes(self):
        """Return dict with fill values for mixed-dtype Dataset variables and coords."""
        ds = xr.Dataset(
            data_vars={
                "fvar": ("x", np.array([1.5, 2.5], dtype=float)),
                "ivar": ("x", np.array([1, 2], dtype=np.int16)),
            },
            coords={"x": np.array([0, 1], dtype=int)},
        )
        ds["ivar"].attrs["_FillValue"] = -7
        result = define_fill_value_dictionary(ds)
        assert set(result.keys()) == {"fvar", "ivar", "x"}
        assert result["fvar"] is dtypes.NA
        assert result["ivar"] == -7
        assert result["x"] == np.iinfo(np.int64).max
