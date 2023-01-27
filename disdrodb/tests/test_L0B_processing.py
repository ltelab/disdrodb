import os
import pytest
import pandas as pd
import numpy as np
import xarray as xr
from disdrodb.L0 import L0B_processing
from disdrodb.L0 import io

PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "pytest_files",
)


def test_check_L0_raw_fields_available():
    # not tested yet because relies on config files that can be modified
    # function_return = L0B_processing.check_L0_raw_fields_available()
    assert 1 == 1


def test_infer_split_str():
    # Test strings with no delimiter
    assert L0B_processing.infer_split_str("") == None
    assert L0B_processing.infer_split_str("abc") == None

    # Test strings with semicolon delimiter
    assert L0B_processing.infer_split_str("a;b;c") == ";"
    assert L0B_processing.infer_split_str("a;b;c;") == ";"

    # Test strings with comma delimiter
    assert L0B_processing.infer_split_str("a,b,c") == ","
    assert L0B_processing.infer_split_str("a,b,c,") == ","

    # Test strings with both semicolon and comma delimiters
    assert L0B_processing.infer_split_str("a;b,c;d,e") == ";"
    assert L0B_processing.infer_split_str("a,b;c,d;e") == ";"


def test_format_string_array():
    # Test empty string
    assert np.allclose(L0B_processing.format_string_array("", 4), [0, 0, 0, 0])

    # Test strings with semicolon delimiter
    assert np.allclose(
        L0B_processing.format_string_array("2;44;22;33", 4), [2, 44, 22, 33]
    )
    assert np.allclose(
        L0B_processing.format_string_array("2;44;22;33;", 4), [2, 44, 22, 33]
    )

    # Test strings with comma delimiter
    assert np.allclose(
        L0B_processing.format_string_array("2,44,22,33", 4), [2, 44, 22, 33]
    )
    assert np.allclose(
        L0B_processing.format_string_array("2,44,22,33,", 4), [2, 44, 22, 33]
    )

    # Test strings with incorrect number of values
    assert np.allclose(
        L0B_processing.format_string_array("2,44,22", 4),
        [np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        L0B_processing.format_string_array("2,44,22,33,44", 4),
        [np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )


def test_reshape_raw_spectrum_to_2D():
    # Test valid input array
    arr = np.arange(12).reshape(2, 6)
    n_bins_dict = {"raw_drop_concentration": 2, "raw_drop_average_velocity": 3}
    n_timesteps = 2

    res_arr = L0B_processing.reshape_raw_spectrum_to_2D(arr, n_bins_dict, n_timesteps)
    expected_arr = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

    assert np.allclose(res_arr, expected_arr)

    # Test invalid input array
    arr = np.arange(12).reshape(2, 6)
    n_bins_dict = {"raw_drop_concentration": 2, "raw_drop_average_velocity": 3}
    n_timesteps = 4
    with pytest.raises(ValueError):
        L0B_processing.reshape_raw_spectrum_to_2D(arr, n_bins_dict, n_timesteps)


def test_retrieve_L0B_arrays():
    # not tested yet because relies on config files that can be modified
    # function_return = L0B_processing.retrieve_L0B_arrays()
    assert 1 == 1


def test_get_coords():
    # not tested yet because relies on config files that can be modified
    # function_return = L0B_processing.get_coords()
    assert 1 == 1


def test_convert_object_variables_to_string():

    # Create test dataset
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ds = xr.Dataset.from_dataframe(df)

    # Check that variable 'b' is of type object
    assert pd.api.types.is_object_dtype(ds["b"])

    # Convert variables with object dtype to string
    ds = L0B_processing.convert_object_variables_to_string(ds)

    # Check that variable 'b' is not of type object
    assert not pd.api.types.is_object_dtype(ds["b"])

    # Check that variable 'b' is of type 'string'
    assert pd.api.types.is_string_dtype(ds["b"])

    # Create an xarray Dataset with a variable 'b' of type 'float'
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ds = xr.Dataset.from_dataframe(df)

    # Convert variables with object dtype to string
    ds = L0B_processing.convert_object_variables_to_string(ds)

    # Check that variable 'b' is of type 'float'
    assert ds["b"].dtype == "float"


def test_create_L0B_from_L0A():
    # not tested yet because relies on config files that can be modified
    # function_return = L0B_processing.create_L0B_from_L0A()
    assert 1 == 1


def test_set_variable_attributes():
    # not tested yet because relies on config files that can be modified
    # function_return = L0B_processing.set_variable_attributes()
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

    result = L0B_processing.sanitize_encodings_dict(encoding_dict_1, ds)

    assert isinstance(result, dict)

    # Test that the dictionary contains the same keys as the input dictionary
    assert set(result.keys()) == set(encoding_dict_1.keys())

    # Test that the chunk sizes in the returned dictionary are smaller than or equal to the corresponding array shapes in the dataset
    for var in result.keys():
        assert tuple(result[var]["chunksizes"]) <= ds[var].shape

    result = L0B_processing.sanitize_encodings_dict(encoding_dict_2, ds)

    assert isinstance(result, dict)

    # Test that the dictionary contains the same keys as the input dictionary
    assert set(result.keys()) == set(encoding_dict_2.keys())

    # Test that the chunk sizes in the returned dictionary are smaller than or equal to the corresponding array shapes in the dataset
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
    ds_rechunked = L0B_processing.rechunk_dataset(ds, encoding_dict)
    assert ds_rechunked["a"].chunks == ((1, 1), (2, 1))
    assert ds_rechunked["b"].chunks == ((2,), (1, 1, 1))


def test_set_encodings():
    # not tested yet because relies on config files that can be modified
    # function_return = L0B_processing.set_encodings()
    assert 1 == 1


def test_write_L0B():
    # not tested yet because relies on config files that can be modified
    # function_return = L0B_processing.write_L0B()
    assert 1 == 1


def test_create_L0B_summary():
    # not tested yet because relies on config files that can be modified
    # function_return = L0B_processing.create_L0B_summary()
    assert 1 == 1
