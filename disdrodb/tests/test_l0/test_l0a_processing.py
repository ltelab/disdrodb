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
"""Test DISDRODB L0A processing routines."""

import os

import numpy as np
import pandas as pd
import pytest

from disdrodb.l0 import io, l0a_processing
from disdrodb.l0.l0a_processing import (
    _check_df_sanitizer_fun,
    _check_matching_column_number,
    _check_not_empty_dataframe,
    cast_column_dtypes,
    coerce_corrupted_values_to_nan,
    read_raw_files,
    remove_corrupted_rows,
    remove_issue_timesteps,
    replace_nan_flags,
    set_nan_invalid_values,
    set_nan_outside_data_range,
    strip_delimiter_from_raw_arrays,
    strip_string_spaces,
)

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - create_test_config_files  # defined in tests/conftest.py


raw_data_format_dict = {
    "key_1": {
        "valid_values": [1, 2, 3],
        "data_range": [0, 4],
        "nan_flags": None,
    },
    "key_2": {
        "valid_values": [1, 2, 3],
        "data_range": [0, 89],
        "nan_flags": -9999,
    },
}
config_dict = {"raw_data_format.yml": raw_data_format_dict}


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_set_nan_invalid_values(create_test_config_files):
    """Create a dummy config file and test the function set_nan_invalid_values.

    Parameters
    ----------
    create_test_config_files : function
        Function that creates and removes the dummy config file.
    """
    # Test without modification
    df = pd.DataFrame({"key_1": [1, 2, 1, 2, 1]})
    output = set_nan_invalid_values(df, "test", verbose=False)
    assert df.equals(output)

    # Test with modification
    df = pd.DataFrame({"key_1": [1, 2, 1, 2, 4]})
    output = set_nan_invalid_values(df, "test", verbose=False)
    assert np.isnan(output["key_1"][4])


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_set_nan_outside_data_range(create_test_config_files):
    # Test case 1: Check if the function sets values outside the data range to NaN
    data = {"key_1": [1, 2, 3, 4, 5], "key_2": [0.1, 0.3, 0.5, 0.7, 0.2]}

    df = pd.DataFrame(data)

    result_df = set_nan_outside_data_range(df, "test", verbose=False)

    assert np.isnan(result_df["key_1"][4])


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_replace_nan_flags(create_test_config_files):
    # Create a sample dataframe with nan flags
    data = {
        "key_1": [6, 7, 1, 9, -9999],
        "key_2": [6, 7, 1, 9, -9999],
    }
    df = pd.DataFrame(data)

    # Call the function with the sample dataframe
    sensor_name = "test"
    verbose = True
    df = replace_nan_flags(df, sensor_name, verbose)

    expected_data = {
        "key_1": [6, 7, 1, 9, -9999],
        "key_2": [6, 7, 1, 9, np.nan],
    }
    assert df.equals(pd.DataFrame(expected_data))


def test_remove_corrupted_rows():
    data = {
        "raw_drop_number": ["1", "2", "3", "a", "5"],
        "raw_drop_concentration": ["0.1", "0.3", "0.5", "b", "0.2"],
        "raw_drop_average_velocity": ["2.1", "1.2", "1.8", "c", "2.0"],
    }

    data = pd.DataFrame(data)
    output = remove_corrupted_rows(data)
    assert output.shape[1] == 3

    # Test case 1: Check if the function removes corrupted rows
    data = {
        "raw_drop_number": ["1", "2", "3", "a", "5"],
        "other": ["0.1", "0.3", "0.5", "b", "0.2"],
        "raw_drop_average_velocity": ["2.1", "1.2", "1.8", "c", "2.0"],
    }

    data = pd.DataFrame(data)
    output = remove_corrupted_rows(data)
    assert output.shape[1] == 3

    # Test case 2: Check if the function raises ValueError when there are no remaining rows
    with pytest.raises(ValueError, match=r"No remaining rows after data corruption checks."):
        remove_corrupted_rows(pd.DataFrame())

    # Test case 3: Check if the function raises ValueError when only one row remains
    with pytest.raises(ValueError, match=r"Only 1 row remains after data corruption checks. Check the file."):
        remove_corrupted_rows(pd.DataFrame({"raw_drop_number": ["1"]}))


def test_strip_delimiter_from_raw_arrays():
    data = {"raw_drop_number": ["  value1", "value2 ", "value3  "], "key_3": [" value4", "value5", "value6"]}
    df = pd.DataFrame(data)
    result = strip_delimiter_from_raw_arrays(df)
    expected_data = {"raw_drop_number": ["value1", "value2", "value3"], "key_3": [" value4", "value5", "value6"]}
    expected = pd.DataFrame(expected_data)

    # Check if result matches expected result
    assert result.equals(expected)


l0a_encoding_dict = {
    "key_1": "str",
    "key_2": "int",
    "key_3": "str",
    "key_4": "int64",
}
config_dict = {"l0a_encodings.yml": l0a_encoding_dict}


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_strip_string_spaces(create_test_config_files):
    data = {"key_1": ["  value1", "value2 ", "value3  "], "key_2": [1, 2, 3], "key_3": ["value4", "value5", "value6"]}
    df = pd.DataFrame(data)

    # Call function
    result = strip_string_spaces(df, "test")

    # Define expected result
    expected_data = {
        "key_1": ["value1", "value2", "value3"],
        "key_2": [1, 2, 3],
        "key_3": ["value4", "value5", "value6"],
    }
    expected = pd.DataFrame(expected_data)

    # Check if result matches expected result
    assert result.equals(expected)


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_coerce_corrupted_values_to_nan(create_test_config_files):
    # Test with a valid dataframe
    df = pd.DataFrame({"key_4": ["1"]})
    df_out = coerce_corrupted_values_to_nan(df, "test", verbose=False)

    assert df.equals(df_out)

    # Test with a wrong dataframe
    df = pd.DataFrame({"key_4": ["text"]})
    df_out = coerce_corrupted_values_to_nan(df, "test", verbose=False)
    assert pd.isnull(df_out["key_4"][0])


def test_remove_issue_timesteps():
    # Create dummy dataframe
    df = pd.DataFrame({"time": [1, 2, 3, 4, 5], "col1": [0, 1, 2, 3, 4]})

    # Create dummy issue dictionary with timesteps to remove
    issue_dict = {"timesteps": [2, 4]}

    # Call function to remove problematic timesteps
    df_cleaned = remove_issue_timesteps(df, issue_dict)

    # Check that problematic timesteps were removed
    assert set(df_cleaned["time"]) == {1, 3, 5}


def test__preprocess_reader_kwargs():
    # Test that the function removes the 'dtype' key from the reader_kwargs dict
    reader_kwargs = {"dtype": "int64", "other_key": "other_value", "delimiter": ","}
    preprocessed_kwargs = l0a_processing._preprocess_reader_kwargs(reader_kwargs)
    assert "dtype" not in preprocessed_kwargs
    assert "other_key" in preprocessed_kwargs

    # Test that the function removes the 'blocksize' and 'assume_missing' keys
    # - This argument expected by dask.dataframe.read_csv
    reader_kwargs = {
        "blocksize": 128,
        "assume_missing": True,
        "other_key": "other_value",
        "delimiter": ",",
    }
    preprocessed_kwargs = l0a_processing._preprocess_reader_kwargs(
        reader_kwargs,
    )
    assert "blocksize" not in preprocessed_kwargs
    assert "assume_missing" not in preprocessed_kwargs
    assert "other_key" in preprocessed_kwargs

    # Test raise error if delimiter is not specified
    reader_kwargs = {"dtype": "int64", "other_key": "other_value"}
    with pytest.raises(ValueError):
        l0a_processing._preprocess_reader_kwargs(reader_kwargs)


def test_concatenate_dataframe():
    # Test that the function returns a Pandas dataframe
    df1 = pd.DataFrame({"time": [1, 2, 3], "value": [4, 5, 6]})
    df2 = pd.DataFrame({"time": [7, 8, 9], "value": [10, 11, 12]})
    concatenated_df = l0a_processing.concatenate_dataframe([df1, df2])
    assert isinstance(concatenated_df, pd.DataFrame)

    # Test that the function raises a ValueError if the list_df is empty
    with pytest.raises(ValueError, match="No objects to concatenate"):
        l0a_processing.concatenate_dataframe([])

    with pytest.raises(ValueError):
        l0a_processing.concatenate_dataframe(["not a dataframe"])

    with pytest.raises(ValueError):
        l0a_processing.concatenate_dataframe(["not a dataframe", "not a dataframe"])


def test_strip_delimiter():
    # Test it strips all external  delimiters
    s = ",,,,,"
    assert l0a_processing._strip_delimiter(s) == ""
    s = "0000,00,"
    assert l0a_processing._strip_delimiter(s) == "0000,00"
    s = ",0000,00,"
    assert l0a_processing._strip_delimiter(s) == "0000,00"
    s = ",,,0000,00,,"
    assert l0a_processing._strip_delimiter(s) == "0000,00"
    # Test if empty string, return the empty string
    s = ""
    assert l0a_processing._strip_delimiter(s) == ""
    # Test if None returns None
    s = None
    assert isinstance(l0a_processing._strip_delimiter(s), type(None))
    # Test if np.nan returns np.nan
    s = np.nan
    assert np.isnan(l0a_processing._strip_delimiter(s))


def test_is_not_corrupted():
    # Test empty string
    s = ""
    assert l0a_processing._is_not_corrupted(s)
    # Test valid string (convertible to numeric, after split by ,)
    s = "000,001,000"
    assert l0a_processing._is_not_corrupted(s)
    # Test corrupted string (not convertible to numeric, after split by ,)
    s = "000,xa,000"
    assert not l0a_processing._is_not_corrupted(s)
    # Test None is considered corrupted
    s = None
    assert not l0a_processing._is_not_corrupted(s)
    # Test np.nan is considered corrupted
    s = np.nan
    assert not l0a_processing._is_not_corrupted(s)


def test_cast_column_dtypes():
    # Create a test dataframe with object columns
    df = pd.DataFrame(
        {
            "time": ["2022-01-01 00:00:00", "2022-01-01 00:05:00", "2022-01-01 00:10:00"],
            "station_number": "station_number",
            "altitude": "8849",
        }
    )
    # Call the function
    sensor_name = "OTT_Parsivel"
    df_out = cast_column_dtypes(df, sensor_name, verbose=False)
    # Check that the output dataframe has the correct column types
    assert str(df_out["time"].dtype) == "datetime64[s]"
    assert str(df_out["station_number"].dtype) == "object"
    assert str(df_out["altitude"].dtype) == "float64"


def test_remove_rows_with_missing_time():
    # Create dataframe
    n_rows = 3
    time = pd.date_range(start="2023-01-01", periods=n_rows, freq="D")
    df = pd.DataFrame({"time": time})

    # Add Nat value to a single rows of the time column
    df.at[0, "time"] = np.datetime64("NaT")
    # Test it remove the invalid timestep
    valid_df = l0a_processing.remove_rows_with_missing_time(df)
    assert len(valid_df) == n_rows - 1
    assert not np.any(valid_df["time"].isna())

    # Add only Nat value
    df["time"] = np.repeat([np.datetime64("NaT")], n_rows).astype("M8[s]")

    # Test it raise an error if no valid timesteps left
    with pytest.raises(ValueError):
        l0a_processing.remove_rows_with_missing_time(df=df)


def test_check_not_empty_dataframe():
    # Test with empty dataframe
    with pytest.raises(ValueError) as excinfo:
        _check_not_empty_dataframe(pd.DataFrame())
    assert "The file is empty and has been skipped." in str(excinfo.value)

    # Test with non-empty dataframe
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert _check_not_empty_dataframe(df) is None


def test_check_matching_column_number():
    # Test with a matching number of columns
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert _check_matching_column_number(df, ["A", "B"]) is None

    # Test with a non-matching number of columns
    with pytest.raises(ValueError) as excinfo:
        _check_matching_column_number(df, ["A"])
    assert "The dataframe has 2 columns, while 1 are expected !" in str(excinfo.value)


def test_remove_duplicated_timesteps():
    # Create dataframe
    n_rows = 3
    time = pd.date_range(start="2023-01-01", periods=n_rows, freq="D")
    df = pd.DataFrame({"time": time})

    # Add duplicated timestep value
    df.at[0, "time"] = df["time"][1]

    # Test it removes the duplicated timesteps
    valid_df = l0a_processing.remove_duplicated_timesteps(df=df)
    assert len(valid_df) == n_rows - 1
    assert len(np.unique(valid_df)) == len(valid_df)


def test_drop_timesteps():
    # Number last timesteps to drop
    n = 2
    # Create an array of datetime values for the time column
    # - Add also a NaT
    time = pd.date_range(start="2023-01-01 00:00:00", end="2023-01-01 01:00:00", freq="1 min").to_numpy()
    time[0] = np.datetime64("NaT")
    # Create a random array for the dummy column
    dummy = np.random.rand(len(time) - n)
    # Create the dataframe with the two columns
    df = pd.DataFrame({"time": time[:-n], "dummy": dummy})

    # Define timesteps to drop
    # - One inside, n-1 outside
    timesteps = time[-(n + 1) :]

    # Remove timesteps
    df_out = l0a_processing.drop_timesteps(df, timesteps)

    # Test np.NaT is conserved
    assert np.isnan(df_out["time"])[0]

    # Test all timesteps were dropped
    assert not np.any(df_out["time"].isin(timesteps))

    # Test error is raised if all timesteps are dropped
    with pytest.raises(ValueError):
        l0a_processing.drop_timesteps(df, timesteps=time)


def test_drop_time_periods():
    # Create an array of datetime values for the time column
    time = pd.date_range(start="2023-01-01 00:00:00", end="2023-01-01 01:00:00", freq="1 min").to_numpy()

    # Define inside time_periods
    inside_time_periods = [time[[10, 20]]]

    # Define outside time periods
    outside_time_period = [[np.datetime64("2022-12-01 00:00:00"), np.datetime64("2022-12-20 00:00:00")]]

    # Define time_period removing all data
    full_time_period = [time[[0, len(time) - 1]]]

    # Create the dataframe with the two columns
    dummy = np.random.rand(len(time))
    df = pd.DataFrame({"time": time, "dummy": dummy})

    # Test outside time_periods
    df_out = l0a_processing.drop_time_periods(df, time_periods=outside_time_period)
    pd.testing.assert_frame_equal(df_out, df)

    # Test inside time_periods
    df_out = l0a_processing.drop_time_periods(df, time_periods=inside_time_periods)
    assert not np.any(df_out["time"].between(inside_time_periods[0][0], inside_time_periods[0][1], inclusive="both"))

    # Test raise error if all rows are discarded
    with pytest.raises(ValueError):
        l0a_processing.drop_time_periods(df, time_periods=full_time_period)

    # Test code do not break if all rows are removed after first time_period iteration
    # --> Would raise IndexError otherwise
    time_periods = [full_time_period[0], [inside_time_periods[0]]]
    with pytest.raises(ValueError):
        l0a_processing.drop_time_periods(df, time_periods=time_periods)


def create_fake_csv(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def test_read_raw_data(tmp_path):
    # Create a valid test file
    path_test_data = os.path.join(tmp_path, "test.csv")
    data = {"att_1": ["11", "21"], "att_2": ["12", "22"]}
    create_fake_csv(path_test_data, data)

    reader_kwargs = {}
    reader_kwargs["delimiter"] = ","
    reader_kwargs["header"] = 0
    reader_kwargs["engine"] = "python"

    r = l0a_processing.read_raw_data(
        filepath=path_test_data,
        column_names=["att_1", "att_2"],
        reader_kwargs=reader_kwargs,
    )
    expected_output = pd.DataFrame(data)
    assert r.equals(expected_output)

    # Test with an empty file without column
    path_test_data = os.path.join(tmp_path, "test_empty.csv")
    print(path_test_data)
    data = {}
    create_fake_csv(path_test_data, data)

    # Call the function and catch the exception
    with pytest.raises(UnboundLocalError):
        r = l0a_processing.read_raw_data(
            filepath=path_test_data,
            column_names=[],
            reader_kwargs=reader_kwargs,
        )

    # Test with an empty file with column
    path_test_data = os.path.join(tmp_path, "test_empty2.csv")
    print(path_test_data)
    data = {"att_1": [], "att_2": []}
    create_fake_csv(path_test_data, data)

    # Call the function and catch the exception
    r = l0a_processing.read_raw_data(
        filepath=path_test_data,
        column_names=["att_1", "att_2"],
        reader_kwargs=reader_kwargs,
    )

    # Check that an empty dataframe is returned
    assert r.empty is True


def test_check_df_sanitizer_fun():
    # Test with valid df_sanitizer_fun
    def df_sanitizer_fun(df):
        return df

    assert _check_df_sanitizer_fun(df_sanitizer_fun) is None

    # Test with None argument
    assert _check_df_sanitizer_fun(None) is None

    # Test with non-callable argument
    with pytest.raises(ValueError, match="'df_sanitizer_fun' must be a function."):
        _check_df_sanitizer_fun(123)

    # Test with argument that has more than one parameter
    def bad_fun(x, y):
        pass

    with pytest.raises(ValueError, match="The `df_sanitizer_fun` must have only `df` as input argument!"):
        _check_df_sanitizer_fun(bad_fun)

    # Test with argument that has wrong parameter name
    def bad_fun2(d):
        pass

    with pytest.raises(ValueError, match="The `df_sanitizer_fun` must have only `df` as input argument!"):
        _check_df_sanitizer_fun(bad_fun2)


def test_write_l0a(tmp_path):
    # create dummy dataframe
    data = [{"a": "1", "b": "2", "c": "3"}, {"a": "2", "b": "2", "c": "3"}]
    df = pd.DataFrame(data).set_index("a")
    df["time"] = pd.Timestamp.now()

    # Write parquet file
    path_parquet_file = os.path.join(tmp_path, "fake_data_sample.parquet")
    l0a_processing.write_l0a(df, path_parquet_file, True, False)

    # Read parquet file
    df_written = io.read_l0a_dataframe([path_parquet_file], False)

    # Check if parquet file are similar
    is_equal = df.equals(df_written)

    assert is_equal


def test_read_raw_files():
    # Set up the inputs
    filepaths = ["test_file1.csv", "test_file2.csv"]
    column_names = ["time", "value"]
    reader_kwargs = {"delimiter": ","}
    sensor_name = "my_sensor"
    verbose = False

    # Create a test dataframe
    df1 = pd.DataFrame(
        {"time": pd.date_range(start="2022-01-01", end="2022-01-02", freq="H"), "value": np.random.rand(25)}
    )
    df2 = pd.DataFrame(
        {"time": pd.date_range(start="2022-01-03", end="2022-01-04", freq="H"), "value": np.random.rand(25)}
    )
    df_list = [df1, df2]

    # Mock the process_raw_file function
    # The code block is defining a mock function called mock_process_raw_file
    # which will be used in unit testing to replace the original process_raw_file function.
    def mock_process_raw_file(filepath, column_names, reader_kwargs, df_sanitizer_fun, sensor_name, verbose):
        if filepath == "test_file1.csv":
            return df1
        elif filepath == "test_file2.csv":
            return df2

    # Monkey patch the function
    l0a_processing.process_raw_file = mock_process_raw_file

    # Call the function
    result = read_raw_files(
        filepaths=filepaths,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        sensor_name=sensor_name,
        verbose=verbose,
    )

    # Check the result
    expected_result = pd.concat(df_list).reset_index(drop=True)
    assert result.equals(expected_result)
