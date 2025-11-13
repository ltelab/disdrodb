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
import dask.array
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes

from disdrodb.utils.xarray import (
    define_dataarray_fill_value,
    define_fill_value_dictionary,
    remap_numeric_array,
    remove_diameter_coordinates,
    remove_velocity_coordinates,
    unstack_datarray_dimension,
    xr_get_last_valid_idx,
    xr_remap_numeric_array,
)


class TestUnstackDatarrayDimension:
    def test_invalid_coord_handling_raises(self):
        """It raises ValueError for invalid coord_handling option."""
        da = xr.DataArray([1, 2], dims=["d"], coords={"d": ["a", "b"]}, name="var")
        with pytest.raises(ValueError, match="coord_handling"):
            unstack_datarray_dimension(da, dim="d", coord_handling="invalid")

    def test_coord_handling_keep_retains_coords(self):
        """It keeps non-dimensional coordinates when coord_handling='keep'."""
        da = xr.DataArray(
            [1, 2],
            dims=["d"],
            coords={"d": ["a", "b"], "other": ("d", [10, 20])},
            name="var",
        )
        ds = unstack_datarray_dimension(da, dim="d", coord_handling="keep")
        # "other" coord should remain
        assert "other" in ds.coords

    def test_coord_handling_drop_removes_coords(self):
        """It drops non-dimensional coordinates when coord_handling='drop'."""
        da = xr.DataArray(
            [1, 2],
            dims=["d"],
            coords={"d": ["a", "b"], "other": ("d", [10, 20])},
            name="var",
        )
        ds = unstack_datarray_dimension(da, dim="d", coord_handling="drop")
        # "other" coord should be removed
        assert "other" not in ds.coords

    def test_coord_handling_unstack_splits_coords(self):
        """It unstacks non-dimensional coordinates when coord_handling='unstack'."""
        da = xr.DataArray(
            [1, 2],
            dims=["d"],
            coords={"d": ["a", "b"], "other": ("d", [10, 20])},
            name="var",
        )
        ds = unstack_datarray_dimension(da, dim="d", coord_handling="unstack", prefix="p_", suffix="_")
        # Expect split variables for "var" and "other"
        assert set(ds.data_vars).issuperset({"p_var_a", "p_var_b", "p_other_a", "p_other_b"})

    def test_unstack_datarray_dimension_with_prefix_suffix(self):
        """Test splitting a DataArray with prefix and suffix."""
        da = xr.DataArray(
            np.array([5, 6, 7]),
            dims=("dim1",),
            coords={"dim1": ["X", "Y", "Z"]},
            name="var",
        )
        ds = unstack_datarray_dimension(da, dim="dim1", prefix="split_", suffix="_")
        assert set(ds.data_vars) == {"split_var_X", "split_var_Y", "split_var_Z"}
        np.testing.assert_array_equal(ds["split_var_X"].values, [5])
        np.testing.assert_array_equal(ds["split_var_Y"].values, [6])
        np.testing.assert_array_equal(ds["split_var_Z"].values, [7])


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


class TestRemoveDiameterAndVelocityCoordinates:
    def test_remove_diameter_coordinates(self):
        """It removes all diameter coordinates if present."""
        coords = {
            name: ("x", [1, 2, 3])
            for name in [
                "diameter_bin_center",
                "diameter_bin_width",
                "diameter_bin_lower",
                "diameter_bin_upper",
            ]
        }
        ds = xr.Dataset(coords=coords)
        result = remove_diameter_coordinates(ds)
        for c in ["diameter_bin_center", "diameter_bin_width", "diameter_bin_lower", "diameter_bin_upper"]:
            assert c not in result.coords

    def test_remove_velocity_coordinates(self):
        """It removes all velocity coordinates if present."""
        coords = {
            name: ("x", [1, 2, 3])
            for name in [
                "velocity_bin_center",
                "velocity_bin_width",
                "velocity_bin_lower",
                "velocity_bin_upper",
            ]
        }
        ds = xr.Dataset(coords=coords)
        result = remove_velocity_coordinates(ds)
        for c in ["velocity_bin_center", "velocity_bin_width", "velocity_bin_lower", "velocity_bin_upper"]:
            assert c not in result.coords


class TestRemapNumericArray:
    """Comprehensive tests for remap_numeric_array."""

    # -------------------- NumPy tests -------------------- #

    def test_numpy_basic_remap(self):
        """Verify basic NumPy remapping with dictionary keys and NaN fill for missing values."""
        arr = np.array([1, 2, 3, 4])
        remap = {1: 10, 2: 20, 3: 30}
        result = remap_numeric_array(arr, remap)
        expected = np.array([10, 20, 30, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_numpy_with_fill_value(self):
        """Check NumPy remapping with a custom fill_value for unmapped elements."""
        arr = np.array([1, 5, 3])
        remap = {1: 100, 3: 300}
        result = remap_numeric_array(arr, remap, fill_value=-1)
        expected = np.array([100, -1, 300])
        np.testing.assert_array_equal(result, expected)

    def test_numpy_no_remap_hits(self):
        """Ensure NumPy arrays with no matching keys are filled entirely with NaN."""
        arr = np.array([7, 8, 9])
        remap = {1: 10, 2: 20}
        result = remap_numeric_array(arr, remap)
        expected = np.array([np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_numpy_empty_array(self):
        """Confirm that an empty NumPy array returns an empty array."""
        arr = np.array([])
        remap = {1: 10}
        result = remap_numeric_array(arr, remap)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_numpy_with_nan_values(self):
        """Verify correct handling of NaN and unmapped elements in NumPy arrays."""
        arr = np.array([1, np.nan, 3, 30, 99])
        remap = {1: 10, 3: 30}
        result = remap_numeric_array(arr, remap)
        expected = np.array([10, np.nan, 30, np.nan, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_numpy_mixed_type_remap(self):
        """Test NumPy remapping where target values are floats."""
        arr = np.array([1, 2])
        remap = {1: 10.5, 2: 20.2}
        result = remap_numeric_array(arr, remap)
        expected = np.array([10.5, 20.2])
        np.testing.assert_array_equal(result, expected)

    def test_numpy_inverted_remap(self):
        """Validate NumPy remapping with inverted key-value mapping."""
        arr = np.array([1, 2])
        remap = {1: 2, 2: 1}
        result = remap_numeric_array(arr, remap)
        expected = np.array([2, 1])
        np.testing.assert_array_equal(result, expected)

    # -------------------- Dask tests -------------------- #

    def test_dask_basic_remap(self):
        """Verify basic Dask remapping produces correct computed output."""
        arr = dask.array.from_array(np.array([1, 2, 3, 4]), chunks=2)
        remap = {1: 10, 2: 20, 3: 30}
        result = remap_numeric_array(arr, remap)
        computed = result.compute()
        expected = np.array([10, 20, 30, np.nan])
        np.testing.assert_array_equal(computed, expected)

    def test_dask_with_fill_value(self):
        """Check Dask remapping with a custom fill_value for missing keys."""
        arr = dask.array.from_array(np.array([1, 5, 3]), chunks=2)
        remap = {1: 100, 3: 300}
        result = remap_numeric_array(arr, remap, fill_value=-1)
        computed = result.compute()
        expected = np.array([100, -1, 300])
        np.testing.assert_array_equal(computed, expected)

    def test_dask_with_nan_values(self):
        """Ensure Dask arrays handle NaN and unmapped values properly."""
        arr = dask.array.from_array(np.array([1, np.nan, 3, 99]), chunks=2)
        remap = {1: 10, 3: 30}
        result = remap_numeric_array(arr, remap)
        computed = result.compute()
        expected = np.array([10, np.nan, 30, np.nan])
        np.testing.assert_array_equal(computed, expected)

    def test_dask_dtype_preserved(self):
        """Confirm that Dask dtype remains consistent after remapping."""
        arr = dask.array.from_array(np.array([1, 2, 3], dtype=int), chunks=1)
        remap = {1: 100, 2: 200, 3: 300}
        result = remap_numeric_array(arr, remap)
        assert str(result.dtype).startswith("int") or str(result.dtype).startswith("float")

    # -------------------- Xarray tests -------------------- #

    def test_xarray_basic_remap(self):
        """Verify Xarray DataArray remapping preserves structure and applies correctly."""
        da = xr.DataArray(np.array([1, 2, 3, 4]), dims="x")
        remap = {1: 10, 2: 20, 3: 30}
        result = xr_remap_numeric_array(da, remap)
        assert isinstance(result, xr.DataArray)
        expected = np.array([10, 20, 30, np.nan])
        np.testing.assert_array_equal(result.data, expected)
