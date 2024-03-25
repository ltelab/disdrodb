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
import pytest
import xarray as xr

from disdrodb.l0.l0b_nc_processing import (
    _check_dict_names_validity,
    _get_missing_variables,
    add_dataset_missing_variables,
    rename_dataset,
    replace_custom_nan_flags,
    replace_nan_flags,
    set_nan_invalid_values,
    set_nan_outside_data_range,
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
l0b_encoding_dict = {k: "dummy" for k in mock_valid_names}


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
        }
    )

    # Call the replace_nan_flags function
    result_ds = replace_nan_flags(ds, sensor_name=TEST_SENSOR_NAME, verbose=True)

    # Assertions
    assert result_ds["key_1"].equals(ds["key_1"]), "Key 1 should remain unchanged"
    assert result_ds["key_2"].equals(xr.DataArray([1, np.nan, 2, 3, 89])), "Key 2 nan flags not replaced correctly"
    assert result_ds["key_3"].equals(
        xr.DataArray([1.0, np.nan, 2.0, 3.0, 89.0])
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
        }
    )

    result_ds = set_nan_outside_data_range(ds, TEST_SENSOR_NAME, verbose=True)

    assert result_ds["key_1"].equals(ds["key_1"]), "Key 1 should remain unchanged"
    assert result_ds["key_2"].equals(
        xr.DataArray([np.nan, 10, 50, np.nan, 30])
    ), "Key 2 data range not applied correctly"
    assert result_ds["key_3"].equals(
        xr.DataArray([np.nan, -10, 0, 10, np.nan])
    ), "Key 3 data range not applied correctly"
    assert result_ds["key_4"].equals(ds["key_4"]), "If data_range for key4 is None, data should remain unchanged"
    assert result_ds["key_not_in_dict"].equals(ds["key_not_in_dict"]), "Unrelated keys should remain unchanged"


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_set_nan_invalid_values(create_test_config_files):
    ds = xr.Dataset(
        {
            "key_1": xr.DataArray([0, 1, 2, 3, 4]),
            "key_2": xr.DataArray([9, 10, 20, 30, 40]),
            "key_3": xr.DataArray([0, 0.1, 0.2, 0.3, 1.0]),
            "key_4": xr.DataArray([0, 0, 0, 1, 1]),
            "key_not_in_dict": xr.DataArray([0, 1, 2, 3, 4]),
        }
    )

    result_ds = set_nan_invalid_values(ds, TEST_SENSOR_NAME, verbose=True)

    assert result_ds["key_1"].equals(ds["key_1"]), "Key 1 should remain unchanged"
    assert result_ds["key_2"].equals(
        xr.DataArray([np.nan, np.nan, np.nan, np.nan, np.nan])
    ), "Key 2 valid values not applied correctly"
    assert result_ds["key_3"].equals(
        xr.DataArray([0.0, np.nan, np.nan, np.nan, 1.0])
    ), "Key 3 float values not processed correctly"
    assert result_ds["key_4"].equals(
        xr.DataArray([0, 0, 0, 1, 1])
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
        }
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
        }
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
        }
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
