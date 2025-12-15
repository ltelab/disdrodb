# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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


import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.l0.l0b_processing import (
    add_dataset_crs_coords,
    convert_object_variables_to_string,
    ensure_valid_geolocation,
    format_string_array,
    generate_l0b,
    infer_split_str,
    replace_empty_strings_with_zeros,
    reshape_raw_spectrum,
    retrieve_l0b_arrays,
    set_variable_attributes,
)
from disdrodb.l0.standards import (
    get_bin_coords_dict,
    get_dims_size_dict,
    get_raw_array_dims_order,
)

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
        },
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
        "l0b_cf_attrs.yml": cf_attrs,
    }
    return dummy_configs_dict


@pytest.mark.parametrize("create_test_config_files", [define_test_dummy_configs()], indirect=True)
def test_generate_l0b(create_test_config_files):
    n_times = 10

    def make_repeated_string(n: int, value: str = "000") -> str:
        """Return a comma-separated string like '000,000,...' with n repetitions."""
        return ",".join([value] * n)

    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "time": pd.date_range("2022-01-01", periods=n_times, freq="h"),
            "raw_drop_concentration": [make_repeated_string(32) for _ in range(n_times)],
            "raw_drop_average_velocity": [make_repeated_string(32) for _ in range(n_times)],
            "raw_drop_number": [make_repeated_string(1024) for _ in range(n_times)],
            "latitude": np.random.rand(n_times),
            "longitude": np.random.rand(n_times),
            "altitude": np.random.rand(n_times),
        },
    )
    # Create a sample attrs dictionary
    metadata = {
        "sensor_name": "PARSIVEL",
        "latitude": 46.52130,
        "longitude": 6.56786,
        "altitude": 400,
        "platform_type": "fixed",
    }

    # Call the function
    ds = generate_l0b(df, metadata=metadata)

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
        "raw_drop_concentration",
        "raw_drop_average_velocity",
        "raw_drop_number",
    ]
    assert set(ds.variables) == set(expected_variables)
    assert set(ds.dims) == {"diameter_bin_center", "time", "velocity_bin_center", "crs"}

    # Check that the geolocation coordinates have been properly set
    assert np.allclose(ds["latitude"].to_numpy(), df["latitude"].to_numpy())
    assert np.allclose(ds["longitude"].to_numpy(), df["longitude"].to_numpy())
    assert np.allclose(ds["altitude"].to_numpy(), df["altitude"].to_numpy())

    # Check that the dataset has a CRS coordinate
    assert "crs" in ds.coords

    # Assert that raise error if any raw_* columns present
    df_bad = df.drop(columns=["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number"])
    with pytest.raises(ValueError):
        generate_l0b(df_bad, metadata=metadata)


def test_add_dataset_crs_coords():
    # Create example dataset
    ds = xr.Dataset(
        {
            "var1": xr.DataArray([1, 2, 3], dims="time"),
            "lat": xr.DataArray([0, 1, 2], dims="time"),
            "lon": xr.DataArray([0, 1, 2], dims="time"),
        },
    )

    # Call the function and check the output
    ds_out = add_dataset_crs_coords(ds)
    assert "crs" in ds_out.coords
    assert ds_out["crs"].to_numpy() == "WGS84"


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
def test_get_bin_coords_dict(create_test_config_files):
    result = get_bin_coords_dict("test")
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
        infer_split_str(None)

    # Test strings with no delimiter
    assert infer_split_str("") is None
    assert infer_split_str("") is None
    assert infer_split_str("abc") is None

    # Test strings with semicolon delimiter
    assert infer_split_str("a;b;c") == ";"
    assert infer_split_str("a;b;c;") == ";"

    # Test strings with comma delimiter
    assert infer_split_str("a,b,c") == ","
    assert infer_split_str("a,b,c,") == ","

    # Test strings with both semicolon and comma delimiters
    assert infer_split_str("a;b,c;d;e") == ";"


def test_replace_empty_strings_with_zeros():
    values = np.array(["", "0", "", "1"])
    output = replace_empty_strings_with_zeros(values).tolist()
    expected_output = np.array(["0", "0", "0", "1"]).tolist()
    assert output == expected_output


def test_format_string_array():
    # Tests splitter behaviour with None
    assert [] == []

    # Test empty string
    assert np.allclose(format_string_array("", 4), [0, 0, 0, 0])

    # Test strings with semicolon and column delimiter
    assert np.allclose(format_string_array("2;44;22;33", 4), [2, 44, 22, 33])
    assert np.allclose(format_string_array("2,44,22,33", 4), [2, 44, 22, 33])
    assert np.allclose(format_string_array("000;000;000;001", 4), [0, 0, 0, 1])

    # Test strip away excess delimiters
    assert np.allclose(format_string_array(",,2,44,22,33,,", 4), [2, 44, 22, 33])
    # Test strings with incorrect number of values
    arr_nan = [np.nan, np.nan, np.nan, np.nan]
    assert np.allclose(format_string_array("2,44,22", 4), arr_nan, equal_nan=True)

    assert np.allclose(format_string_array("2,44,22,33,44", 4), arr_nan, equal_nan=True)
    # Test strings with incorrect format
    assert np.allclose(format_string_array(",,2,", 4), arr_nan, equal_nan=True)


def test_reshape_raw_spectrum():
    list_sensor_name = ["LPM", "PARSIVEL"]
    # sensor_name = "LPM"
    # sensor_name = "PARSIVEL"

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
        if sensor_name in ["LPM"]:
            flat_spectrum = expected_spectrum.flatten(order="F")
        elif sensor_name in ["PARSIVEL", "PARSIVEL2"]:
            flat_spectrum = expected_spectrum.flatten(order="C")
        else:
            raise NotImplementedError(f"Unavailable test for {sensor_name}")

        # Create array [time, ...] to mock retrieve_l0b_arrays code
        arr = np.stack([flat_spectrum], axis=0)

        # Now reshape spectrum and check is correct
        dims_order_dict = get_raw_array_dims_order(sensor_name=sensor_name)
        dims_size_dict = get_dims_size_dict(sensor_name=sensor_name)
        dims_order = dims_order_dict["raw_drop_number"]
        arr, dims = reshape_raw_spectrum(
            arr=arr,
            dims_order=dims_order,
            dims_size_dict=dims_size_dict,
            n_timesteps=1,
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
        reshape_raw_spectrum(
            arr=arr,
            dims_order=dims_order,
            dims_size_dict=dims_size_dict,
            n_timesteps=1,
        )


def test_retrieve_l0b_arrays():
    from disdrodb.l0.standards import (
        get_dims_size_dict,
    )

    list_sensor_name = ["LPM", "PARSIVEL"]
    # sensor_name = "LPM"
    # sensor_name = "PARSIVEL"

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
        if sensor_name in ["LPM"]:
            flat_spectrum = expected_spectrum.flatten(order="F")
        elif sensor_name in ["PARSIVEL", "PARSIVEL2"]:
            flat_spectrum = expected_spectrum.flatten(order="C")
        else:
            raise NotImplementedError(f"Unavailable test for {sensor_name}")

        # Create L0A dataframe with single row
        raw_spectrum = ",".join(flat_spectrum)
        df = pd.DataFrame({"dummy": ["row1", "row2"], "raw_drop_number": raw_spectrum})

        # Use retrieve_l0b_arrays
        data_vars = retrieve_l0b_arrays(df=df, sensor_name=sensor_name, verbose=False)
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
    ds = convert_object_variables_to_string(ds)

    # Check that variable 'b' is not of type object
    assert not pd.api.types.is_object_dtype(ds["b"])

    # Check that variable 'b' is of type 'string'
    assert pd.api.types.is_string_dtype(ds["b"])

    # Create an xarray Dataset with a variable 'b' of type 'float'
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ds = xr.Dataset.from_dataframe(df)

    # Convert variables with object dtype to string
    ds = convert_object_variables_to_string(ds)

    # Check that variable 'b' is of type 'float'
    assert ds["b"].dtype == "float"


class TestEnsureValidGeolocation:
    """Unit tests for ensure_valid_geolocation function."""

    def test_altitude_scalar_ignore(self):
        """Test scalar altitude outside range is unchanged when errors='ignore'."""
        ds = xr.Dataset({"altitude": xr.DataArray(-5)})
        result = ensure_valid_geolocation(ds, "altitude", errors="ignore")
        assert result["altitude"].item() == -5  # unchanged

    def test_latitude_scalar_raise(self):
        """Test scalar latitude outside range raises ValueError when errors='raise'."""
        ds = xr.Dataset({"latitude": xr.DataArray(100)})
        with pytest.raises(ValueError, match="latitude out of range"):
            ensure_valid_geolocation(ds, "latitude", errors="raise")

    def test_longitude_scalar_coerce(self):
        """Test scalar longitude outside range is set to NaN when errors='coerce'."""
        ds = xr.Dataset({"longitude": xr.DataArray(200)})
        result = ensure_valid_geolocation(ds, "longitude", errors="coerce")
        assert np.isnan(result["longitude"].item())

    def test_altitude_time_ignore(self):
        """Test time-varying altitude with invalid values is unchanged when errors='ignore'."""
        ds = xr.Dataset({"altitude": ("time", [0, 10, -5, 50])})
        result = ensure_valid_geolocation(ds, "altitude", errors="ignore")
        np.testing.assert_array_equal(result["altitude"].to_numpy(), [0, 10, -5, 50])

    def test_latitude_time_coerce(self):
        """Test time-varying latitude with invalid values coerced to NaN."""
        ds = xr.Dataset({"latitude": ("time", [-100, 0, 45, 120])})
        result = ensure_valid_geolocation(ds, "latitude", errors="coerce")
        vals = result["latitude"].to_numpy()
        assert np.isnan(vals[0])
        assert np.isnan(vals[3])
        assert vals[1] == 0
        assert vals[2] == 45

    def test_longitude_time_raise(self):
        """Test time-varying longitude with invalid values raises ValueError."""
        ds = xr.Dataset({"longitude": ("time", [0, 90, 200, -190])})
        with pytest.raises(ValueError, match="longitude out of range"):
            ensure_valid_geolocation(ds, "longitude", errors="raise")

    def test_invalid_coordinate_name(self):
        """Test function raises ValueError when coordinate is not valid geolocation."""
        ds = xr.Dataset({"lat": xr.DataArray([1000])})
        with pytest.raises(ValueError, match="Valid geolocation coordinates"):
            ensure_valid_geolocation(ds, "lat")
