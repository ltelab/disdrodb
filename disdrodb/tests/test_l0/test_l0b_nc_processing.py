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
"""Test DISDRODB L0B (from raw netCDFs) processing routines."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.l0.l0b_nc_processing import (
    _check_dict_names_validity,
    _get_missing_variables,
    add_dataset_missing_variables,
    remove_issue_timesteps,
    rename_dataset,
    replace_custom_nan_flags,
    replace_nan_flags,
    set_nan_invalid_values,
    set_nan_outside_data_range,
    standardize_raw_dataset,
    subset_dataset,
)

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - create_test_config_files  # defined in tests/conftest.py


raw_data_format_dict = {
    # Nothing should be done
    "key_1": {
        "valid_values": None,
        "data_range": None,
        "nan_flags": None,
    },
    # Set nans
    "key_2": {
        "valid_values": [1, 2, 3],
        "data_range": [10, 50],
        "nan_flags": -9999,
    },
    # Assess possible int/float problems
    "key_3": {
        "valid_values": [0, 1],
        "data_range": [-10, 10],
        "nan_flags": -9999.0,
    },
    # List of nan flags
    "key_4": {
        "valid_values": [0, 1],
        "data_range": None,
        "nan_flags": [-9999, -8888],
    },
}

mock_valid_names = ["var1", "var2", "var3", "var_not_in_ds"]
l0b_encoding_dict = dict.fromkeys(mock_valid_names, "dummy")


config_dict = {"raw_data_format.yml": raw_data_format_dict, "l0b_encodings.yml": l0b_encoding_dict}

TEST_SENSOR_NAME = "test"


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_replace_nan_flags(create_test_config_files):
    """Create a dummy config file and test the function replace_nan_flags.

    Parameters
    ----------
    create_test_config_files : function
        Function that creates and removes the dummy config file.
    """
    # Mock xarray Dataset
    ds = xr.Dataset(
        {
            "key_1": xr.DataArray([0, 1, 2, 3, 4]),
            "key_2": xr.DataArray([1, -9999, 2, 3, 89]),
            "key_3": xr.DataArray([1.0, -9999.0, 2.0, 3.0, 89.0]),
            "key_4": xr.DataArray([1, -9999, -8888, 2, 3]),
            "key_not_in_dict": xr.DataArray([10, 20, 30, 40, 50]),
        },
    )

    # Call the replace_nan_flags function
    result_ds = replace_nan_flags(ds, sensor_name=TEST_SENSOR_NAME, verbose=True)

    # Assertions
    assert result_ds["key_1"].equals(ds["key_1"]), "Key 1 should remain unchanged"
    assert result_ds["key_2"].equals(xr.DataArray([1, np.nan, 2, 3, 89])), "Key 2 nan flags not replaced correctly"
    assert result_ds["key_3"].equals(
        xr.DataArray([1.0, np.nan, 2.0, 3.0, 89.0]),
    ), "Key 3 float values not processed correctly"
    assert result_ds["key_4"].equals(xr.DataArray([1, np.nan, np.nan, 2, 3])), "Key 4 nan flags not replaced correctly"
    assert result_ds["key_not_in_dict"].equals(ds["key_not_in_dict"]), "Unrelated keys should remain unchanged"


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_set_nan_outside_data_range(create_test_config_files):
    ds = xr.Dataset(
        {
            "key_1": xr.DataArray([0, 1, 2, 3, 4]),
            "key_2": xr.DataArray([9, 10, 50, 51, 30]),
            "key_3": xr.DataArray([-11, -10, 0, 10, 11]),
            "key_4": xr.DataArray([99, 100, 150, 200, 201]),
            "key_not_in_dict": xr.DataArray([0, 1, 2, 3, 4]),
        },
    )

    result_ds = set_nan_outside_data_range(ds, TEST_SENSOR_NAME, verbose=True)

    assert result_ds["key_1"].equals(ds["key_1"]), "Key 1 should remain unchanged"
    assert result_ds["key_2"].equals(
        xr.DataArray([np.nan, 10, 50, np.nan, 30]),
    ), "Key 2 data range not applied correctly"
    assert result_ds["key_3"].equals(
        xr.DataArray([np.nan, -10, 0, 10, np.nan]),
    ), "Key 3 data range not applied correctly"
    assert result_ds["key_4"].equals(ds["key_4"]), "If data_range for key4 is None, data should remain unchanged"
    assert result_ds["key_not_in_dict"].equals(ds["key_not_in_dict"]), "Unrelated keys should remain unchanged"


class TestRemoveIssueTimesteps:

    def test_no_issues_returns_same(self):
        """Should return identical dataset when no issues provided."""
        times = pd.date_range("2025-01-01T00:00", periods=10, freq="1min")
        ds = xr.Dataset({"value": ("time", range(10))}, coords={"time": times})
        ds_out = remove_issue_timesteps(ds, issue_dict={})
        assert ds_out.sizes["time"] == ds.sizes["time"]
        pd.testing.assert_index_equal(ds_out["time"].to_index(), ds["time"].to_index())

    def test_drop_timesteps(self):
        """Should drop specific timesteps from the dataset."""
        timesteps = pd.date_range("2025-01-01T00:00", periods=10, freq="1min")
        ds = xr.Dataset({"value": ("time", range(10))}, coords={"time": timesteps})
        # Pick two times to drop and define issue dict
        timesteps_to_drop = timesteps.to_numpy()[[2, 5]]
        timesteps_to_drop_str = timesteps_to_drop.astype("M8[s]").astype(str).tolist()
        issue_dict = {"timesteps": timesteps_to_drop_str}

        # Test remaining timesteps
        ds_filtered = remove_issue_timesteps(ds, issue_dict=issue_dict)
        assert all(t not in timesteps_to_drop for t in ds_filtered["time"].to_numpy())
        assert ds_filtered.sizes["time"] == 8

    def test_drop_time_periods(self):
        """Should drop all timesteps within given periods."""
        timesteps = pd.date_range("2025-01-01T00:00", periods=10, freq="1min")
        ds = xr.Dataset({"value": ("time", range(10))}, coords={"time": timesteps})
        # Define time period from index 3 to 6 inclusive
        timesteps_str = timesteps.to_numpy().astype("M8[s]").astype(str)
        start_time = timesteps.to_numpy()[3]
        end_time = timesteps.to_numpy()[6]
        start_time_str = timesteps_str[3]
        end_time_str = timesteps_str[6]
        issue_dict = {"time_periods": [(start_time_str, end_time_str)]}
        # Test remaining timesteps
        ds_filtered = remove_issue_timesteps(ds, issue_dict=issue_dict)
        remaining_timesteps = ds_filtered["time"].to_numpy()
        assert all((t < start_time) or (t > end_time) for t in remaining_timesteps)
        assert ds_filtered.sizes["time"] == 6

    def test_combined_issues(self):
        """Should apply both timesteps and periods removal."""
        timesteps = pd.date_range("2025-01-01T00:00", periods=10, freq="1min")
        ds = xr.Dataset({"value": ("time", range(10))}, coords={"time": timesteps})

        # Define issue dict
        drop_ts = [ds["time"].to_numpy()[1], ds["time"].to_numpy()[8]]
        start = ds["time"].to_numpy()[4]
        end = ds["time"].to_numpy()[5]
        issue_dict = {"timesteps": drop_ts, "time_periods": [(start, end)]}

        # Test remaining timesteps
        ds_filtered = remove_issue_timesteps(ds, issue_dict=issue_dict)
        remaining_timesteps = ds_filtered["time"].to_numpy()
        # ensure none of the drop_ts or period times remain
        for t in drop_ts:
            assert t not in remaining_timesteps
        assert all((t < start) or (t > end) for t in remaining_timesteps)
        # expected size = original 10 - 2 timesteps - 2 period entries = 6
        assert ds_filtered.sizes["time"] == 6

    def test_raises_error_if_all_timesteps_are_removed(self):
        """Should raise ValueError when all timesteps are removed."""
        timesteps = pd.date_range("2025-01-01T00:00", periods=10, freq="1min")
        ds = xr.Dataset({"value": ("time", range(10))}, coords={"time": timesteps})

        # Define issue dict that covers entire time period
        start = ds["time"].to_numpy()[0]
        end = ds["time"].to_numpy()[-1]
        issue_dict = {"time_periods": [(start, end)]}

        # Test raise error
        with pytest.raises(ValueError, match="No timesteps left after removing problematic"):
            remove_issue_timesteps(ds, issue_dict=issue_dict)

        # Define issue dict with timesteps matching all dataset timesteps
        issue_dict = {"timesteps": ds["time"].to_numpy()}
        # Test raise error
        with pytest.raises(ValueError, match="No timesteps left after removing problematic"):
            remove_issue_timesteps(ds, issue_dict=issue_dict)


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_set_nan_invalid_values(create_test_config_files):
    ds = xr.Dataset(
        {
            "key_1": xr.DataArray([0, 1, 2, 3, 4]),
            "key_2": xr.DataArray([9, 10, 20, 30, 40]),
            "key_3": xr.DataArray([0, 0.1, 0.2, 0.3, 1.0]),
            "key_4": xr.DataArray([0, 0, 0, 1, 1]),
            "key_not_in_dict": xr.DataArray([0, 1, 2, 3, 4]),
        },
    )

    result_ds = set_nan_invalid_values(ds, TEST_SENSOR_NAME, verbose=True)

    assert result_ds["key_1"].equals(ds["key_1"]), "Key 1 should remain unchanged"
    assert result_ds["key_2"].equals(
        xr.DataArray([np.nan, np.nan, np.nan, np.nan, np.nan]),
    ), "Key 2 valid values not applied correctly"
    assert result_ds["key_3"].equals(
        xr.DataArray([0.0, np.nan, np.nan, np.nan, 1.0]),
    ), "Key 3 float values not processed correctly"
    assert result_ds["key_4"].equals(
        xr.DataArray([0, 0, 0, 1, 1]),
    ), "Key 4 should not have been modified. Only valid values are present."
    assert result_ds["key_not_in_dict"].equals(ds["key_not_in_dict"]), "Unrelated keys should remain unchanged"


def test_replace_custom_nan_flags():
    # Custom dictionary of nan flags for testing
    dict_nan_flags = {"key_1": [-999], "key_2": [-9999, -8888], "key_3": [0]}

    # Mock xarray Dataset
    ds = xr.Dataset(
        {
            "key_1": xr.DataArray([1, -999, 2, 3, 4]),
            "key_2": xr.DataArray([1, -9999, -8888, 2, 3]),
            "key_3": xr.DataArray([0, 1, 0, 2, 3]),
            "key_not_in_flags": xr.DataArray([10, 20, 30, 40, 50]),
        },
    )

    # Call the replace_custom_nan_flags function
    result_ds = replace_custom_nan_flags(ds, dict_nan_flags=dict_nan_flags)

    # Assertions
    assert result_ds["key_1"].equals(xr.DataArray([1, np.nan, 2, 3, 4])), "Key 1 nan flags not replaced correctly"
    assert result_ds["key_2"].equals(xr.DataArray([1, np.nan, np.nan, 2, 3])), "Key 2 nan flags not replaced correctly"
    assert result_ds["key_3"].equals(xr.DataArray([np.nan, 1, np.nan, 2, 3])), "Key 3 nan flags not replaced correctly"
    assert result_ds["key_not_in_flags"].equals(ds["key_not_in_flags"]), "Unrelated keys should remain unchanged"


def test_add_dataset_missing_variables(monkeypatch):
    # Mock variables and their dimensions
    mock_var_dims_dict = {"missing_var_1": ["dim1"], "missing_var_2": ["dim1", "dim2"]}

    # Mock get_variables_dimension function
    def mock_get_variables_dimension(sensor_name):
        return mock_var_dims_dict

    monkeypatch.setattr("disdrodb.l0.standards.get_variables_dimension", mock_get_variables_dimension)

    # Define xarray Dataset
    ds = xr.Dataset({"existing_var": xr.DataArray(np.random.rand(5, 3), dims=["dim1", "dim2"])})

    # List of missing variables
    missing_vars = ["missing_var_1", "missing_var_2"]

    # Call add_dataset_missing_variables
    result_ds = add_dataset_missing_variables(ds, missing_vars=missing_vars, sensor_name="sensor_name")

    # Assertions
    assert "missing_var_1" in result_ds, "missing_var_1 should be in the dataset"
    assert "missing_var_2" in result_ds, "missing_var_2 should be in the dataset"
    assert result_ds["missing_var_1"].shape == (5,), "Shape of missing_var_1 is incorrect"
    assert result_ds["missing_var_2"].shape == (5, 3), "Shape of missing_var_2 is incorrect"
    assert np.all(np.isnan(result_ds["missing_var_1"])), "Values of missing_var_1 should be NaN"
    assert np.all(np.isnan(result_ds["missing_var_2"])), "Values of missing_var_2 should be NaN"


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_check_dict_names_validity(create_test_config_files):
    # Define dict_names with valid values
    dict_names_valid = {"key1": "var1", "key2": "var2"}
    # No exception should be raised
    _check_dict_names_validity(dict_names_valid, sensor_name=TEST_SENSOR_NAME)

    # Define dict_names with some invalid values
    dict_names_invalid = {"key1": "invalid_name", "key2": "var2"}
    with pytest.raises(ValueError):
        _check_dict_names_validity(dict_names_invalid, sensor_name=TEST_SENSOR_NAME)


def test_rename_dataset():
    # Define xarray Dataset with variables, coordinates, and dimensions
    ds = xr.Dataset(
        {"var1": (("dim1", "dim2"), np.random.rand(2, 3)), "var2": (("dim1", "dim2"), np.random.rand(2, 3))},
        coords={"dim2": [1, 2, 3], "coord1": ("dim2", [1, 2, 3])},
    )

    # Define dict_names for renaming
    dict_names = {
        "var1": "new_var1",
        "var2": "new_var2",
        "dim1": "new_dim1",
        "dim2": "new_dim2",
        "coord1": "new_coord1",
        "non_existing_var": "should_be_ignored",
    }

    # Call rename_dataset
    result_ds = rename_dataset(ds, dict_names)

    # Assertions
    assert "new_var1" in result_ds, "var1 should be renamed to new_var1"
    assert "new_var2" in result_ds, "var2 should be renamed to new_var2"
    assert "new_dim1" in result_ds.dims, "dim1 should be renamed to new_dim1"
    assert "new_dim2" in result_ds.coords, "dim2 should be renamed to new_dim2"
    assert "new_coord1" in result_ds.coords, "coord1 should be renamed to new_coord1"
    assert "coord1" not in result_ds.coords, "coord1 should be renamed to new_coord1"

    assert "non_existing_var" not in result_ds, "non_existing_var should not be in the renamed dataset"
    assert "var1" not in result_ds, "Original var1 should not exist after renaming"
    assert "var2" not in result_ds, "Original var2 should not exist after renaming"


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_subset_dataset(create_test_config_files):
    # Define xarray Dataset with extra variables (assumed to be renamed)
    ds = xr.Dataset(
        {
            "var1": xr.DataArray([1, 2, 3]),
            "var2": xr.DataArray([4, 5, 6]),
            "var3": xr.DataArray([7, 8, 9]),
            "var_not_needed": xr.DataArray([10, 11, 12]),
        },
    )

    # Define dict_names mapping
    # - Key are used to rename (the values are used for subsetting)
    # - Values are used for subsetting
    dict_names = {"key1": "var1", "key2": "var2", "key3": "var3", "key4": "var_not_in_ds"}

    # Call subset_dataset
    result_ds = subset_dataset(ds, dict_names=dict_names, sensor_name=TEST_SENSOR_NAME)

    # Assertions
    assert set(result_ds.data_vars) == {"var1", "var2", "var3"}, "Dataset should only contain var1, var2, and var3"
    assert "var_not_needed" not in result_ds, "var_not_needed should not be in the subset dataset"
    assert "var_not_in_ds" not in result_ds, "var_not_in_ds should not be in the subset dataset"


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_get_missing_variables(create_test_config_files):
    # Define xarray Dataset with some variables (assumed to be renamed and subsetted)
    ds = xr.Dataset(
        {
            "var1": xr.DataArray([1, 2, 3]),
            "var2": xr.DataArray([4, 5, 6]),
        },
    )

    # Define dict_names mapping
    # - Key are used to rename (the values are used for subsetting)
    # - Values are used for subsetting
    dict_names = {"key1": "var1", "key2": "var2", "key3": "var3", "key4": "var_not_in_ds"}

    # Call _get_missing_variables
    ds = rename_dataset(ds=ds, dict_names=dict_names)
    missing_vars = _get_missing_variables(ds, dict_names, sensor_name=TEST_SENSOR_NAME)

    # Assertions
    assert missing_vars == {"var3", "var_not_in_ds"}, "Missing variables should be identified correctly"


class TestStandardizeRawDataset:
    def create_test_dataset(self, include_raw_data=True, include_optional_vars=True):
        """Create a test dataset based on DELFT_NC dict_names structure."""
        # Create time coordinate
        times = pd.date_range("2023-01-01", periods=5, freq="60s")

        # Create diameter and velocity classes (32 bins each for PARSIVEL2)
        diameter_classes = np.arange(32)
        velocity_classes = np.arange(32)

        # Create the base dataset structure
        data_vars = {}
        coords = {
            "time": times,
            "diameter_classes": diameter_classes,
            "velocity_classes": velocity_classes,
        }

        # Add required variables
        if include_raw_data:
            data_vars["raw_data"] = (
                ("time", "diameter_classes", "velocity_classes"),
                np.random.randint(0, 100, size=(5, 32, 32)),
            )

        if include_optional_vars:
            data_vars.update(
                {
                    "rainfall_rate_32bit": (("time",), np.random.rand(5) * 10),
                    "fieldV": (("time", "velocity_classes"), np.random.rand(5, 32) * 20),
                    "fieldN": (("time", "diameter_classes"), np.random.rand(5, 32) * 1000),
                },
            )

        return xr.Dataset(data_vars, coords=coords)

    def get_dict_names(self):
        """Get the dict_names mapping for testing."""
        return {
            ### Dimensions
            "diameter_classes": "diameter_bin_center",
            "velocity_classes": "velocity_bin_center",
            ### Variables
            "rainfall_rate_32bit": "rainfall_rate_32bit",
            "fieldV": "raw_drop_average_velocity",
            "fieldN": "raw_drop_concentration",
            "raw_data": "raw_drop_number",
        }

    def test_missing_raw_drop_number_raises_error(self):
        """Test that missing raw_drop_number raises ValueError."""
        # Create dataset without raw_data
        ds = self.create_test_dataset(include_raw_data=False, include_optional_vars=True)
        dict_names = self.get_dict_names()

        with pytest.raises(ValueError, match="The raw drop spectrum is not present in the netCDF file!"):
            standardize_raw_dataset(ds, dict_names, "PARSIVEL2")

    def test_successful_processing_all_variables_present(self):
        """Test successful processing when all variables are present."""
        ds = self.create_test_dataset(include_raw_data=True, include_optional_vars=True)
        dict_names = self.get_dict_names()

        result = standardize_raw_dataset(ds, dict_names, "PARSIVEL2")

        # Check that all variables were renamed correctly
        assert "raw_drop_number" in result.data_vars
        assert "raw_drop_average_velocity" in result.data_vars
        assert "raw_drop_concentration" in result.data_vars
        assert "rainfall_rate_32bit" in result.data_vars

        # Check that old names are gone
        assert "raw_data" not in result.data_vars
        assert "fieldV" not in result.data_vars
        assert "fieldN" not in result.data_vars

        # Check dimensions were renamed
        assert "diameter_bin_center" in result.dims
        assert "velocity_bin_center" in result.dims
        assert "diameter_classes" not in result.dims
        assert "velocity_classes" not in result.dims

        # Check coordinates were added
        assert "diameter_bin_width" in result.coords
        assert "diameter_bin_lower" in result.coords
        assert "diameter_bin_upper" in result.coords
        assert "velocity_bin_width" in result.coords
        assert "velocity_bin_lower" in result.coords
        assert "velocity_bin_upper" in result.coords

        # Check data shapes are preserved
        assert result["raw_drop_number"].shape == (5, 32, 32)
        assert result["raw_drop_average_velocity"].shape == (5, 32)
        assert result["raw_drop_concentration"].shape == (5, 32)
        assert result["rainfall_rate_32bit"].shape == (5,)

        # Check dimensions are correct
        assert result["raw_drop_number"].dims == ("time", "diameter_bin_center", "velocity_bin_center")
        assert result["raw_drop_average_velocity"].dims == ("time", "velocity_bin_center")
        assert result["raw_drop_concentration"].dims == ("time", "diameter_bin_center")
        assert result["rainfall_rate_32bit"].dims == ("time",)

    def test_missing_optional_variables_infilled_with_nan(self):
        """Test that missing optional variables are infilled with NaN."""
        # Create dataset with only raw_data
        ds = self.create_test_dataset(include_raw_data=True, include_optional_vars=False)
        dict_names = self.get_dict_names()

        result = standardize_raw_dataset(ds, dict_names, "PARSIVEL2")

        # Check that raw_drop_number is present and not NaN
        assert "raw_drop_number" in result.data_vars
        assert not np.all(np.isnan(result["raw_drop_number"].values))

        # Check that missing variables were added as NaN
        expected_missing = ["raw_drop_average_velocity", "raw_drop_concentration", "rainfall_rate_32bit"]
        for var in expected_missing:
            assert var in result.data_vars
            assert np.all(np.isnan(result[var].values))

        # Check shapes of infilled variables
        assert result["raw_drop_average_velocity"].sizes == {"diameter_bin_center": 32, "time": 5}
        assert result["raw_drop_concentration"].sizes == {"diameter_bin_center": 32, "time": 5}
        assert result["rainfall_rate_32bit"].shape == (5,)

    def test_partial_missing_variables(self):
        """Test with some variables missing."""
        # Create dataset with raw_data and only fieldV
        ds = self.create_test_dataset(include_raw_data=True, include_optional_vars=False)
        ds["fieldV"] = (("time", "velocity_classes"), np.random.rand(5, 32) * 20)

        dict_names = self.get_dict_names()

        result = standardize_raw_dataset(ds, dict_names, "PARSIVEL2")

        # Check that present variables have real data
        assert "raw_drop_number" in result.data_vars
        assert "raw_drop_average_velocity" in result.data_vars
        assert not np.all(np.isnan(result["raw_drop_number"].values))
        assert not np.all(np.isnan(result["raw_drop_average_velocity"].values))

        # Check that missing variables were infilled
        missing_vars = ["raw_drop_concentration", "rainfall_rate_32bit"]
        for var in missing_vars:
            assert var in result.data_vars
            assert np.all(np.isnan(result[var].values))

    def test_coordinates_updated_with_bin_coords(self):
        """Test that bin coordinates are properly updated."""
        ds = self.create_test_dataset(include_raw_data=True, include_optional_vars=True)
        dict_names = self.get_dict_names()

        result = standardize_raw_dataset(ds, dict_names, "PARSIVEL2")

        # Check that coordinate values were updated (should be different from simple arange)
        # The actual bin coordinates should come from get_bin_coords_dict("PARSIVEL2")
        assert "diameter_bin_center" in result.coords
        assert "velocity_bin_center" in result.coords

        # Check that coordinates have the right length
        assert len(result.coords["diameter_bin_center"]) == 32
        assert len(result.coords["velocity_bin_center"]) == 32

    def test_data_integrity_preserved(self):
        """Test that data values are preserved during the transformation."""
        ds = self.create_test_dataset(include_raw_data=True, include_optional_vars=True)
        dict_names = self.get_dict_names()

        # Store original values
        original_raw_data = ds["raw_data"].to_numpy()
        original_fieldV = ds["fieldV"].to_numpy()
        original_fieldN = ds["fieldN"].to_numpy()
        original_rainfall = ds["rainfall_rate_32bit"].to_numpy()

        result = standardize_raw_dataset(ds, dict_names, "PARSIVEL2")

        # Check that data values are preserved
        np.testing.assert_array_equal(result["raw_drop_number"].values, original_raw_data)
        np.testing.assert_array_equal(result["raw_drop_average_velocity"].values, original_fieldV)
        np.testing.assert_array_equal(result["raw_drop_concentration"].values, original_fieldN)
        np.testing.assert_array_equal(result["rainfall_rate_32bit"].values, original_rainfall)

    def test_invalid_sensor_name(self):
        """Test that invalid sensor name raises appropriate error."""
        ds = self.create_test_dataset(include_raw_data=True, include_optional_vars=True)
        dict_names = self.get_dict_names()

        with pytest.raises(ValueError):
            standardize_raw_dataset(ds, dict_names, "INVALID_SENSOR")

    def test_subset_behavior_with_extra_variables(self):
        """Test that extra variables not in dict_names are properly handled."""
        ds = self.create_test_dataset(include_raw_data=True, include_optional_vars=True)

        # Add an extra variable that's not in dict_names
        ds["extra_variable"] = (("time",), np.random.rand(5))

        dict_names = self.get_dict_names()

        result = standardize_raw_dataset(ds, dict_names, "PARSIVEL2")

        # Extra variable should not be in the result (assuming it's not a valid DISDRODB variable)
        assert "extra_variable" not in result.data_vars

        # But expected variables should still be there
        assert "raw_drop_number" in result.data_vars
        assert "raw_drop_average_velocity" in result.data_vars
