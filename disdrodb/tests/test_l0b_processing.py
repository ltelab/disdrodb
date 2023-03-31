import os
import pytest
import pandas as pd
import numpy as np
import xarray as xr
from disdrodb.l0 import l0b_processing

PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "pytest_files",
)


def test_infer_split_str():
    # Test type eror if string=None
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
        get_raw_array_dims_order,
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

    # Test unvalid inputs
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


def test_get_bin_coords():
    # not tested yet because relies on config files that can be modified
    # function_return = l0b_processing.get_bin_coords()
    assert 1 == 1


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


def test_create_l0b_from_l0a():
    # not tested yet because relies on config files that can be modified
    # function_return = l0b_processing.create_l0b_from_l0a()
    assert 1 == 1


def test_set_variable_attributes():
    # not tested yet because relies on config files that can be modified
    # function_return = l0b_processing.set_variable_attributes()
    assert 1 == 1


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
