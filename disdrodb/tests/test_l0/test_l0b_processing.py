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
"""Test DISDRODB L0B processing routines."""


import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb import __root_path__
from disdrodb.l0 import l0b_processing
from disdrodb.l0.l0b_processing import (
    _set_attrs_dict,
    add_dataset_crs_coords,
    create_l0b_from_l0a,
    get_bin_coords,
    set_coordinate_attributes,
    set_variable_attributes,
)

PATH_TEST_FOLDERS_FILES = os.path.join(__root_path__, "disdrodb", "tests", "data")


# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - create_test_config_files  # defined in tests/conftest.py


def define_test_dummy_configs():
    """Define a dictionary with dummy configuration files."""
    raw_data_format_dict = {
        "rainfall_rate_32bit": {
            "n_digits": 7,
            "n_characters": 8,
            "n_decimals": 3,
            "n_naturals": 4,
            "data_range": [0, 9999.999],
            "nan_flags": None,
        }
    }
    bins_velocity_dict = {
        "center": {0: 0.05, 1: 0.15, 2: 0.25, 3: 0.35, 4: 0.45},
        "bounds": {0: [0.0, 0.1], 1: [0.1, 0.2], 2: [0.2, 0.3], 3: [0.3, 0.4], 4: [0.4, 0.5]},
        "width": {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
    }
    bins_diameter_dict = {
        "center": {0: 0.062, 1: 0.187, 2: 0.312, 3: 0.437, 4: 0.562},
        "bounds": {
            0: [0.0, 0.1245],
            1: [0.1245, 0.2495],
            2: [0.2495, 0.3745],
            3: [0.3745, 0.4995],
            4: [0.4995, 0.6245],
        },
        "width": {0: 0.125, 1: 0.125, 2: 0.125, 3: 0.125, 4: 0.125},
    }
    cf_attrs = {
        "raw_drop_concentration": {
            "units": "1/(m3*mm)",
            "description": "Particle number concentrations per diameter class",
            "long_name": "Raw drop concentration",
        },
        "raw_drop_average_velocity": {
            "units": "m/s",
            "description": "Average particle velocities for each diameter class",
            "long_name": "Raw drop average velocity",
        },
        "raw_drop_number": {
            "units": "",
            "description": "Drop counts per diameter and velocity class",
            "long_name": "Raw drop number",
        },
    }
    dummy_configs_dict = {
        "raw_data_format.yml": raw_data_format_dict,
        "bins_velocity.yml": bins_velocity_dict,
        "bins_diameter.yml": bins_diameter_dict,
        "l0b_variables_attrs.yml": cf_attrs,
    }
    return dummy_configs_dict


@pytest.mark.parametrize("create_test_config_files", [define_test_dummy_configs()], indirect=True)
def test_create_l0b_from_l0a(create_test_config_files):
    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "time": pd.date_range("2022-01-01", periods=10, freq="H"),
            "raw_drop_concentration": np.random.rand(10),
            "raw_drop_average_velocity": np.random.rand(10),
            "raw_drop_number": np.random.rand(10),
            "latitude": np.random.rand(10),
            "longitude": np.random.rand(10),
            "altitude": np.random.rand(10),
        }
    )
    # Create a sample attrs dictionary
    attrs = {
        "sensor_name": "test",
        "latitude": 46.52130,
        "longitude": 6.56786,
        "altitude": 400,
        "platform_type": "fixed",
    }

    # Call the function
    ds = create_l0b_from_l0a(df, attrs)

    # Check the output dataset has the correct variables and dimensions
    expected_variables = [
        "diameter_bin_lower",
        "latitude",
        "velocity_bin_upper",
        "velocity_bin_width",
        "velocity_bin_center",
        "velocity_bin_lower",
        "time",
        "crs",
        "altitude",
        "longitude",
        "diameter_bin_upper",
        "diameter_bin_width",
        "diameter_bin_center",
    ]
    assert set(ds.variables) == set(expected_variables)
    assert set(ds.dims) == set(["diameter_bin_center", "time", "velocity_bin_center", "crs"])

    # Check that the geolocation coordinates have been properly set
    assert np.allclose(ds.latitude.values, df.latitude.values)
    assert np.allclose(ds.longitude.values, df.longitude.values)
    assert np.allclose(ds.altitude.values, df.altitude.values)

    # Check that the dataset has a CRS coordinate
    assert "crs" in ds.coords


def test_add_dataset_crs_coords():
    # Create example dataset
    ds = xr.Dataset(
        {
            "var1": xr.DataArray([1, 2, 3], dims="time"),
            "lat": xr.DataArray([0, 1, 2], dims="time"),
            "lon": xr.DataArray([0, 1, 2], dims="time"),
        }
    )

    # Call the function and check the output
    ds_out = add_dataset_crs_coords(ds)
    assert "crs" in ds_out.coords
    assert ds_out.crs.values == "WGS84"


def test_set_attrs_dict():
    ds = xr.Dataset({"var1": xr.DataArray([1, 2, 3], dims="time")})
    attrs_dict = {"var1": {"attr1": "value1"}}
    ds = _set_attrs_dict(ds, attrs_dict)
    assert ds.var1.attrs["attr1"] == "value1"

    attrs_dict = {"var2": {"attr1": "value1"}}
    ds = _set_attrs_dict(ds, attrs_dict)
    assert "var2" not in ds

    attrs_dict = {"var1": {"attr1": "value1"}, "var2": {"attr2": "value2"}}
    ds = _set_attrs_dict(ds, attrs_dict)
    assert ds.var1.attrs["attr1"] == "value1"
    assert "var2" not in ds


def test_set_coordinate_attributes():
    # Create example dataset
    ds = xr.Dataset(
        {
            "var1": xr.DataArray([1, 2, 3], dims="time"),
            "lat": xr.DataArray([0, 1, 2], dims="time"),
            "lon": xr.DataArray([0, 1, 2], dims="time"),
        }
    )
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"

    # Call the function and check the output
    ds_out = set_coordinate_attributes(ds)
    assert "units" in ds_out.lat.attrs
    assert ds_out.lat.attrs["units"] == "degrees_north"
    assert "units" in ds_out.lon.attrs
    assert ds_out.lon.attrs["units"] == "degrees_east"
    assert "units" not in ds_out.var1.attrs


def test_set_variable_attributes(mocker):
    # Create a sample dataset
    data = np.random.rand(10, 10)
    ds = xr.Dataset({"var_1": (("lat", "lon"), data)})
    sensor_name = "my_sensor"

    # Create mock functions for attribute dictionaries
    mocked_cf_dict = {
        "var_1": {
            "description": "descrition_1",
            "units": "unit_1",
            "long_name": "long_1",
        },
        "var_2": {
            "description": "descrition_2",
            "units": "unit_2",
            "long_name": "long_2",
        },
    }
    mocker.patch(
        "disdrodb.l0.l0b_processing.get_l0b_cf_attrs_dict",
        return_value=mocked_cf_dict,
    )
    mocker.patch(
        "disdrodb.l0.l0b_processing.get_data_range_dict",
        return_value={"var_1": [0, 1], "var_2": [0, 1]},
    )

    # Call the function to set variable attributes
    ds = set_variable_attributes(ds, sensor_name)
    assert ds["var_1"].attrs["description"] == "descrition_1"
    assert ds["var_1"].attrs["units"] == "unit_1"
    assert ds["var_1"].attrs["long_name"] == "long_1"
    assert ds["var_1"].attrs["valid_min"] == 0
    assert ds["var_1"].attrs["valid_max"] == 1


dummy_configs_dict = define_test_dummy_configs()
config_names = ["bins_diameter.yml", "bins_velocity.yml"]
bins_configs_dicts = {key: dummy_configs_dict[key].copy() for key in config_names}


@pytest.mark.parametrize("create_test_config_files", [bins_configs_dicts], indirect=True)
def test_get_bin_coords(create_test_config_files):
    result = get_bin_coords("test")
    expected_result = {
        "diameter_bin_center": [0.062, 0.187, 0.312, 0.437, 0.562],
        "diameter_bin_lower": (["diameter_bin_center"], [0.0, 0.1245, 0.2495, 0.3745, 0.4995]),
        "diameter_bin_upper": (["diameter_bin_center"], [0.1245, 0.2495, 0.3745, 0.4995, 0.6245]),
        "diameter_bin_width": (["diameter_bin_center"], [0.125, 0.125, 0.125, 0.125, 0.125]),
        "velocity_bin_center": (["velocity_bin_center"], [0.05, 0.15, 0.25, 0.35, 0.45]),
        "velocity_bin_lower": (["velocity_bin_center"], [0.0, 0.1, 0.2, 0.3, 0.4]),
        "velocity_bin_upper": (["velocity_bin_center"], [0.1, 0.2, 0.3, 0.4, 0.5]),
        "velocity_bin_width": (["velocity_bin_center"], [0.1, 0.1, 0.1, 0.1, 0.1]),
    }

    assert result == expected_result


def test_infer_split_str():
    # Test type error if string=None
    with pytest.raises(TypeError):
        l0b_processing.infer_split_str(None)

    # Test strings with no delimiter
    assert l0b_processing.infer_split_str("") is None
    assert l0b_processing.infer_split_str("") is None
    assert l0b_processing.infer_split_str("abc") is None

    # Test strings with semicolon delimiter
    assert l0b_processing.infer_split_str("a;b;c") == ";"
    assert l0b_processing.infer_split_str("a;b;c;") == ";"

    # Test strings with comma delimiter
    assert l0b_processing.infer_split_str("a,b,c") == ","
    assert l0b_processing.infer_split_str("a,b,c,") == ","

    # Test strings with both semicolon and comma delimiters
    assert l0b_processing.infer_split_str("a;b,c;d;e") == ";"


def test_replace_empty_strings_with_zeros():
    values = np.array(["", "0", "", "1"])
    output = l0b_processing._replace_empty_strings_with_zeros(values).tolist()
    expected_output = np.array(["0", "0", "0", "1"]).tolist()
    assert output == expected_output


def test_format_string_array():
    # Tests splitter behaviour with None
    assert "".split(None) == []

    # Test empty string
    assert np.allclose(l0b_processing.format_string_array("", 4), [0, 0, 0, 0])

    # Test strings with semicolon and column delimiter
    assert np.allclose(l0b_processing.format_string_array("2;44;22;33", 4), [2, 44, 22, 33])
    assert np.allclose(l0b_processing.format_string_array("2,44,22,33", 4), [2, 44, 22, 33])
    assert np.allclose(l0b_processing.format_string_array("000;000;000;001", 4), [0, 0, 0, 1])

    # Test strip away excess delimiters
    assert np.allclose(l0b_processing.format_string_array(",,2,44,22,33,,", 4), [2, 44, 22, 33])
    # Test strings with incorrect number of values
    arr_nan = [np.nan, np.nan, np.nan, np.nan]
    assert np.allclose(l0b_processing.format_string_array("2,44,22", 4), arr_nan, equal_nan=True)

    assert np.allclose(l0b_processing.format_string_array("2,44,22,33,44", 4), arr_nan, equal_nan=True)
    # Test strings with incorrect format
    assert np.allclose(l0b_processing.format_string_array(",,2,", 4), arr_nan, equal_nan=True)


def test_reshape_raw_spectrum():
    from disdrodb.l0.standards import (
        get_dims_size_dict,
        get_raw_array_dims_order,
    )

    list_sensor_name = ["Thies_LPM", "OTT_Parsivel"]
    # sensor_name = "Thies_LPM"
    # sensor_name = "OTT_Parsivel"

    for sensor_name in list_sensor_name:
        # Retrieve number of bins
        dims_size_dict = get_dims_size_dict(sensor_name=sensor_name)
        n_diameter_bins = dims_size_dict["diameter_bin_center"]
        n_velocity_bins = dims_size_dict["velocity_bin_center"]

        # Define expected spectrum
        # --> row: velocity bins, columns : diameter bins
        expected_spectrum = np.zeros((n_velocity_bins, n_diameter_bins)).astype(str)
        for i in range(0, n_velocity_bins):
            for j in range(0, n_diameter_bins):
                expected_spectrum[i, j] = f"v{i}d{j}"  # v{velocity_bin}d{diameter_bin}
                # expected_spectrum[i, j] = f"{i}.{j}"  # {velocity_bin}.{diameter_bin}

        da_expected_spectrum = xr.DataArray(data=expected_spectrum, dims=["velocity_bin_center", "diameter_bin_center"])

        # Define flattened raw spectrum
        # - OTT: first all diameters bins for velocity bin 1, ...
        # - Thies: first al velocity bins for diameter bin 1, ...
        if sensor_name in ["Thies_LPM"]:
            flat_spectrum = expected_spectrum.flatten(order="F")
        elif sensor_name in ["OTT_Parsivel", "OTT_Parsivel2"]:
            flat_spectrum = expected_spectrum.flatten(order="C")
        else:
            raise NotImplementedError(f"Unavailable test for {sensor_name}")

        # Create array [time, ...] to mock retrieve_l0b_arrays code
        arr = np.stack([flat_spectrum], axis=0)

        # Now reshape spectrum and check is correct
        dims_order_dict = get_raw_array_dims_order(sensor_name=sensor_name)
        dims_size_dict = get_dims_size_dict(sensor_name=sensor_name)
        dims_order = dims_order_dict["raw_drop_number"]
        arr, dims = l0b_processing.reshape_raw_spectrum(
            arr=arr, dims_order=dims_order, dims_size_dict=dims_size_dict, n_timesteps=1
        )
        # Create DataArray and enforce same dimension order as da_expected_spectrum
        da = xr.DataArray(data=arr, dims=dims)
        da = da.isel(time=0)
        da = da.transpose("velocity_bin_center", "diameter_bin_center")

        # Check reshape correctness
        assert da.isel({"velocity_bin_center": 10, "diameter_bin_center": 5}).data.item() == "v10d5"

        # Check value correctness
        xr.testing.assert_equal(da, da_expected_spectrum)

    # Test invalid inputs
    dims_size_dict["diameter_bin_center"] = 20
    with pytest.raises(ValueError):
        l0b_processing.reshape_raw_spectrum(
            arr=arr, dims_order=dims_order, dims_size_dict=dims_size_dict, n_timesteps=1
        )


def test_retrieve_l0b_arrays():
    from disdrodb.l0.standards import (
        get_dims_size_dict,
    )

    list_sensor_name = ["Thies_LPM", "OTT_Parsivel"]
    # sensor_name = "Thies_LPM"
    # sensor_name = "OTT_Parsivel"

    for sensor_name in list_sensor_name:
        # Retrieve number of bins
        dims_size_dict = get_dims_size_dict(sensor_name=sensor_name)
        n_diameter_bins = dims_size_dict["diameter_bin_center"]
        n_velocity_bins = dims_size_dict["velocity_bin_center"]

        # Define expected spectrum
        # --> row: velocity bins, columns : diameter bins
        expected_spectrum = np.zeros((n_velocity_bins, n_diameter_bins)).astype(str)
        for i in range(0, n_velocity_bins):
            for j in range(0, n_diameter_bins):
                expected_spectrum[i, j] = f"{i}.{j}"  # v{velocity_bin}.{diameter_bin}

        da_expected_spectrum = xr.DataArray(
            data=expected_spectrum.astype(float),
            dims=["velocity_bin_center", "diameter_bin_center"],
        )

        # Define flattened raw spectrum
        # - OTT: first all diameters bins for velocity bin 1, ...
        # - Thies: first al velocity bins for diameter bin 1, ...
        if sensor_name in ["Thies_LPM"]:
            flat_spectrum = expected_spectrum.flatten(order="F")
        elif sensor_name in ["OTT_Parsivel", "OTT_Parsivel2"]:
            flat_spectrum = expected_spectrum.flatten(order="C")
        else:
            raise NotImplementedError(f"Unavailable test for {sensor_name}")

        # Create L0A dataframe with single row
        raw_spectrum = ",".join(flat_spectrum)
        df = pd.DataFrame({"dummy": ["row1", "row2"], "raw_drop_number": raw_spectrum})

        # Use retrieve_l0b_arrays
        data_vars = l0b_processing.retrieve_l0b_arrays(df=df, sensor_name=sensor_name, verbose=False)
        # Create Dataset
        ds = xr.Dataset(data_vars=data_vars)

        # Retrieve DataArray
        da = ds["raw_drop_number"].isel(time=0)

        # Enforce same dimension order and type as da_expected_spectrum
        da = da.transpose("velocity_bin_center", "diameter_bin_center").astype(float)

        # Check reshape correctness
        assert da.isel({"velocity_bin_center": 10, "diameter_bin_center": 5}).data.item() == 10.5

        # Check value correctness
        xr.testing.assert_equal(da, da_expected_spectrum)


def test_convert_object_variables_to_string():
    # Create test dataset
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds = xr.Dataset.from_dataframe(df)

    # Check that variable 'b' is of type object
    assert pd.api.types.is_object_dtype(ds["b"])

    # Convert variables with object dtype to string
    ds = l0b_processing.convert_object_variables_to_string(ds)

    # Check that variable 'b' is not of type object
    assert not pd.api.types.is_object_dtype(ds["b"])

    # Check that variable 'b' is of type 'string'
    assert pd.api.types.is_string_dtype(ds["b"])

    # Create an xarray Dataset with a variable 'b' of type 'float'
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ds = xr.Dataset.from_dataframe(df)

    # Convert variables with object dtype to string
    ds = l0b_processing.convert_object_variables_to_string(ds)

    # Check that variable 'b' is of type 'float'
    assert ds["b"].dtype == "float"


@pytest.fixture
def encoding_dict_1():
    # create a test encoding dictionary
    return {
        "var1": {"dtype": "float32", "chunksizes": (10, 10, 10)},
        "var2": {"dtype": "int16", "chunksizes": (5, 5, 5)},
        "var3": {"dtype": "float64", "chunksizes": (100, 100, 100)},
    }


@pytest.fixture
def encoding_dict_2():
    # create a test encoding dictionary
    return {
        "var1": {"dtype": "float32", "chunksizes": (100, 100, 100)},
        "var2": {"dtype": "int16", "chunksizes": (100, 100, 100)},
        "var3": {"dtype": "float64", "chunksizes": (100, 100, 100)},
    }


@pytest.fixture
def ds():
    # create a test xr.Dataset
    data = {
        "var1": (["time", "x", "y"], np.random.random((10, 20, 30))),
        "var2": (["time", "x", "y"], np.random.randint(0, 10, size=(10, 20, 30))),
        "var3": (["time", "x", "y"], np.random.random((10, 20, 30))),
    }
    coords = {"time": np.arange(10), "x": np.arange(20), "y": np.arange(30)}
    return xr.Dataset(data, coords)


def test_sanitize_encodings_dict(encoding_dict_1, encoding_dict_2, ds):
    result = l0b_processing.sanitize_encodings_dict(encoding_dict_1, ds)

    assert isinstance(result, dict)

    # Test that the dictionary contains the same keys as the input dictionary
    assert set(result.keys()) == set(encoding_dict_1.keys())

    # Test that the chunk sizes in the returned dictionary are smaller than or equal to the corresponding array shapes
    # in the dataset
    for var in result.keys():
        assert tuple(result[var]["chunksizes"]) <= ds[var].shape

    result = l0b_processing.sanitize_encodings_dict(encoding_dict_2, ds)

    assert isinstance(result, dict)

    # Test that the dictionary contains the same keys as the input dictionary
    assert set(result.keys()) == set(encoding_dict_2.keys())

    # Test that the chunk sizes in the returned dictionary are smaller than or equal to the corresponding array shapes
    # in the dataset
    for var in result.keys():
        assert tuple(result[var]["chunksizes"]) <= ds[var].shape


def test_rechunk_dataset():
    # Create a sample xarray dataset
    data = {
        "a": (["x", "y"], [[1, 2, 3], [4, 5, 6]]),
        "b": (["x", "y"], [[7, 8, 9], [10, 11, 12]]),
    }
    coords = {"x": [0, 1], "y": [0, 1, 2]}
    ds = xr.Dataset(data, coords=coords)

    # Define the encoding dictionary
    encoding_dict = {"a": {"chunksizes": (1, 2)}, "b": {"chunksizes": (2, 1)}}

    # Test the rechunk_dataset function
    ds_rechunked = l0b_processing.rechunk_dataset(ds, encoding_dict)
    assert ds_rechunked["a"].chunks == ((1, 1), (2, 1))
    assert ds_rechunked["b"].chunks == ((2,), (1, 1, 1))


def test_set_encodings():
    # not tested yet because relies on config files that can be modified
    # function_return = l0b_processing.set_encodings()
    assert 1 == 1


def test_write_l0b():
    # not tested yet because relies on config files that can be modified
    # function_return = l0b_processing.write_l0b()
    assert 1 == 1
