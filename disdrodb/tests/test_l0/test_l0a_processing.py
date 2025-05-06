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

from disdrodb.l0.l0a_processing import (
    cast_column_dtypes,
    check_matching_column_number,
    coerce_corrupted_values_to_nan,
    concatenate_dataframe,
    drop_time_periods,
    drop_timesteps,
    is_raw_array_string_not_corrupted,
    preprocess_reader_kwargs,
    read_l0a_dataframe,
    read_raw_text_file,
    read_raw_text_files,
    remove_corrupted_rows,
    remove_duplicated_timesteps,
    remove_issue_timesteps,
    remove_rows_with_missing_time,
    replace_nan_flags,
    set_nan_invalid_values,
    set_nan_outside_data_range,
    strip_delimiter,
    strip_delimiter_from_raw_arrays,
    strip_string_spaces,
    write_l0a,
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

TEST_SENSOR_NAME = "test"


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
    output = set_nan_invalid_values(df, sensor_name=TEST_SENSOR_NAME, verbose=False)
    assert df.equals(output)

    # Test with modification
    df = pd.DataFrame({"key_1": [1, 2, 1, 2, 4]})
    output = set_nan_invalid_values(df, sensor_name=TEST_SENSOR_NAME, verbose=False)
    assert np.isnan(output["key_1"][4])


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_set_nan_outside_data_range(create_test_config_files):
    """Test that values outside the allowed data range are converted to NaN."""
    # Test case 1: Check if the function sets values outside the data range to NaN
    data = {"key_1": [1, 2, 3, 4, 5], "key_2": [0.1, 0.3, 0.5, 0.7, 0.2]}

    df = pd.DataFrame(data)

    result_df = set_nan_outside_data_range(df, sensor_name=TEST_SENSOR_NAME, verbose=False)

    assert np.isnan(result_df["key_1"][4])


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_replace_nan_flags(create_test_config_files):
    """Test that specified nan_flags are replaced by NaN in the dataframe."""
    # Create a sample dataframe with nan flags
    data = {
        "key_1": [6, 7, 1, 9, -9999],
        "key_2": [6, 7, 1, 9, -9999],
    }
    df = pd.DataFrame(data)

    # Call the function with the sample dataframe
    df = replace_nan_flags(df, sensor_name=TEST_SENSOR_NAME, verbose=True)

    expected_data = {
        "key_1": [6, 7, 1, 9, -9999],
        "key_2": [6, 7, 1, 9, np.nan],
    }
    assert df.equals(pd.DataFrame(expected_data))


class TestRemoveCorruptedRows:
    def test_remove_corrupted_rows_with_all_raw_columns(self):
        """Rows with non-numeric raw-array entries are dropped."""
        data = {
            "raw_drop_number": ["1", "2", "3", "a", "5"],
            "raw_drop_concentration": ["0.1", "0.3", "0.5", "b", "0.2"],
            "raw_drop_average_velocity": ["a", "1.2", "1.8", "c", "2.0"],
        }
        df = pd.DataFrame(data)
        output = remove_corrupted_rows(df)
        # Only the 4 fully numeric rows remain
        assert output.shape == (3, 3)

    def test_remove_corrupted_rows_with_mixed_columns(self):
        """Dropping works even if some columns are not raw arrays."""
        data = {
            "raw_drop_number": ["1", "2", "3", "a", "5"],
            "other": ["0.1", "a", "c", "b", "0.2"],
            "raw_drop_average_velocity": ["2.1", "1.2", "1.8", "c", "2.0"],
        }
        df = pd.DataFrame(data)
        output = remove_corrupted_rows(df)
        # Non-raw 'other' column is still kept, but corrupted rows are removed
        assert output.shape == (4, 3)

    def test_no_rows_remaining_raises(self):
        """ValueError if no rows remain after corruption filtering."""
        with pytest.raises(ValueError, match=r"No remaining rows after data corruption checks."):
            remove_corrupted_rows(pd.DataFrame())

    def test_single_row_remaining_raises(self):
        """ValueError if only one row remains after corruption filtering."""
        df = pd.DataFrame({"raw_drop_number": ["1"]})
        msg = r"Only 1 row remains after data corruption checks. Check the raw file and maybe delete it."
        with pytest.raises(ValueError, match=msg):
            remove_corrupted_rows(df)


def test_strip_delimiter_from_raw_arrays():
    """Test that leading/trailing delimiters are stripped only from raw arrays."""
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


class TestStripStringSpaces:
    @pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
    def test_removes_whitespace_from_string_columns(self, create_test_config_files):
        """Test that whitespace is removed from string columns per encoding config."""
        data = {
            "key_1": ["  value1", "value2 ", "value3  "],
            "key_2": [1, 2, 3],
            "key_3": ["value4", "value5", "value6"],
        }
        df = pd.DataFrame(data)
        result = strip_string_spaces(df, sensor_name=TEST_SENSOR_NAME)
        expected_data = {
            "key_1": ["value1", "value2", "value3"],
            "key_2": [1, 2, 3],
            "key_3": ["value4", "value5", "value6"],
        }
        expected = pd.DataFrame(expected_data)
        assert result.equals(expected)

    @pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
    def test_raises_error_for_non_string_column(self, create_test_config_files):
        """Assert AttributeError when a configured string column is not of string dtype."""
        data = {
            "key_1": [1, 2, 3],
            "key_2": [1, 2, 3],
            "key_3": ["value4", "value5", "value6"],
        }
        df = pd.DataFrame(data)
        with pytest.raises(AttributeError):
            strip_string_spaces(df, sensor_name=TEST_SENSOR_NAME)


@pytest.mark.parametrize("create_test_config_files", [config_dict], indirect=True)
def test_coerce_corrupted_values_to_nan(create_test_config_files):
    """Test that non-convertible strings in numeric columns become NaN."""
    # Test with a valid dataframe
    df = pd.DataFrame({"key_4": ["1"]})
    df_out = coerce_corrupted_values_to_nan(df, sensor_name=TEST_SENSOR_NAME)

    assert df.equals(df_out)

    # Test with a wrong dataframe
    df = pd.DataFrame({"key_4": ["text"]})
    df_out = coerce_corrupted_values_to_nan(df, sensor_name=TEST_SENSOR_NAME)
    assert pd.isna(df_out["key_4"][0])


class RemoveIssueTimesteps:
    def test_remove_issue_timesteps(self):
        """Test removal of timesteps listed in the issue dictionary."""
        # Create dummy dataframe
        df = pd.DataFrame({"time": [1, 2, 3, 4, 5], "col1": [0, 1, 2, 3, 4]})

        # Create dummy issue dictionary with timesteps to remove
        issue_dict = {"timesteps": [2, 4]}

        # Call function to remove problematic timesteps
        df_cleaned = remove_issue_timesteps(df, issue_dict)

        # Check that problematic timesteps were removed
        assert set(df_cleaned["time"]) == {1, 3, 5}

    def test_remove_issue_time_periods(self):
        """Test removal of time ranges specified in the issue dictionary."""
        # Create an array of datetime values for the time column
        timesteps = pd.date_range(start="2023-01-01 00:00:00", end="2023-01-01 01:00:00", freq="1 min").to_numpy()

        # Define issue timesteps and time_periods
        issue_time_periods = [timesteps[[10, 20]]]
        issue_timesteps = timesteps[10:20]

        # Create dummy issue dictionary with timesteps to remove
        issue_dict = {"time_periods": issue_time_periods}

        # Create the dataframe with the two columns
        dummy = np.random.rand(len(timesteps))
        df = pd.DataFrame({"time": timesteps, "dummy": dummy})

        # Call function to remove problematic time_periods
        df_cleaned = remove_issue_timesteps(df, issue_dict)
        assert np.all(~df_cleaned["time"].isin(issue_timesteps))


def test_preprocess_reader_kwargs():
    """Test cleanup of unsupported kwargs and required presence of delimiter."""
    # Test that the function removes the 'dtype' key from the reader_kwargs dict
    reader_kwargs = {"dtype": "int64", "other_key": "other_value", "delimiter": ","}
    preprocessed_kwargs = preprocess_reader_kwargs(reader_kwargs)
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
    preprocessed_kwargs = preprocess_reader_kwargs(
        reader_kwargs,
    )
    assert "blocksize" not in preprocessed_kwargs
    assert "assume_missing" not in preprocessed_kwargs
    assert "other_key" in preprocessed_kwargs

    # Test raise error if delimiter is not specified
    reader_kwargs = {"dtype": "int64", "other_key": "other_value"}
    with pytest.raises(ValueError):
        preprocess_reader_kwargs(reader_kwargs)


def test_concatenate_dataframe():
    """Test concatenation behavior and error handling for invalid inputs."""
    # Test that the function returns a Pandas dataframe
    df1 = pd.DataFrame({"time": [1, 2, 3], "value": [4, 5, 6]})
    df2 = pd.DataFrame({"time": [7, 8, 9], "value": [10, 11, 12]})
    concatenated_df = concatenate_dataframe([df1, df2])
    assert isinstance(concatenated_df, pd.DataFrame)

    # Test that the function raises a ValueError if the list_df is empty
    with pytest.raises(ValueError, match="No objects to concatenate"):
        concatenate_dataframe([])

    with pytest.raises(ValueError):
        concatenate_dataframe(["not a dataframe"])

    with pytest.raises(ValueError):
        concatenate_dataframe(["not a dataframe", "not a dataframe"])


class TestStripDelimiter:
    def test_strips_external_delimiters(self):
        """Test that leading/trailing delimiters are stripped."""
        # strips all leading/trailing commas
        assert strip_delimiter(",,,,,") == ""
        assert strip_delimiter("0000,00,") == "0000,00"
        assert strip_delimiter(",0000,00,") == "0000,00"
        assert strip_delimiter(",,,0000,00,,") == "0000,00"

    def test_empty_string_returns_empty(self):
        """Test that empty strings return empty strings."""
        assert strip_delimiter("") == ""

    def test_none_returns_none(self):
        """Test that None returns None."""
        result = strip_delimiter(None)
        assert result is None

    def test_nan_returns_nan(self):
        """Test that NaN returns NaN."""
        result = strip_delimiter(np.nan)
        assert np.isnan(result)


class TestIsRawArrayStringNotCorrupted:
    def test_empty_string(self):
        """Test that empty strings are not corrupted."""
        s = ""
        assert is_raw_array_string_not_corrupted(s)

    def test_valid_string(self):
        """Test that valid strings are not corrupted."""
        s = "000,001,000"
        assert is_raw_array_string_not_corrupted(s)

    def test_corrupted_string(self):
        """Test that corrupted strings are detected."""
        s = "000,xa,000"
        assert not is_raw_array_string_not_corrupted(s)

    def test_none_is_corrupted(self):
        """Test that None is treated as corrupted."""
        s = None
        assert not is_raw_array_string_not_corrupted(s)

    def test_nan_is_corrupted(self):
        """Test that NaN is treated as corrupted."""
        s = np.nan
        assert not is_raw_array_string_not_corrupted(s)


def test_cast_column_dtypes():
    """Test casting of dataframe columns based on sensor dtype configuration."""
    # Create a test dataframe with object columns
    df = pd.DataFrame(
        {
            "time": ["2022-01-01 00:00:00", "2022-01-01 00:05:00", "2022-01-01 00:10:00"],
            "station_number": "station_number",
            "altitude": "8849",
        },
    )
    # Call the function
    sensor_name = "OTT_Parsivel"
    df_out = cast_column_dtypes(df, sensor_name)
    # Check that the output dataframe has the correct column types
    assert str(df_out["time"].dtype) == "datetime64[s]"
    assert str(df_out["station_number"].dtype) == "object"
    assert str(df_out["altitude"].dtype) == "float64"

    # Assert raise error if can not cast
    df["altitude"] = "text"
    with pytest.raises(ValueError):
        cast_column_dtypes(df, sensor_name)


def test_remove_rows_with_missing_time():
    """Test removal of NaT timestamps and error on all-missing time column."""
    # Create dataframe
    n_rows = 3
    time = pd.date_range(start="2023-01-01", periods=n_rows, freq="D")
    df = pd.DataFrame({"time": time})

    # Add Nat value to a single rows of the time column
    df.loc[0, "time"] = np.datetime64("NaT")
    # Test it remove the invalid timestep
    valid_df = remove_rows_with_missing_time(df)
    assert len(valid_df) == n_rows - 1
    assert not np.any(valid_df["time"].isna())

    # Add only Nat value
    df["time"] = np.repeat([np.datetime64("NaT")], n_rows).astype("M8[s]")

    # Test it raise an error if no valid timesteps left
    with pytest.raises(ValueError):
        remove_rows_with_missing_time(df=df)


def test_check_matching_column_number():
    """Test validation of dataframe column count against expected list."""
    # Test with a matching number of columns
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert check_matching_column_number(df, ["A", "B"]) is None

    # Test with a non-matching number of columns
    with pytest.raises(ValueError) as excinfo:
        check_matching_column_number(df, ["A"])
    assert "The dataframe has 2 columns, while 1 are expected !" in str(excinfo.value)


def test_remove_duplicated_timesteps():
    """Test dropping of duplicated time entries from the dataframe."""
    # Create dataframe
    n_rows = 3
    time = pd.date_range(start="2023-01-01", periods=n_rows, freq="D")
    df = pd.DataFrame({"time": time})

    # Add duplicated timestep value
    df.loc[0, "time"] = df["time"][1]

    # Test it removes the duplicated timesteps
    valid_df = remove_duplicated_timesteps(df=df)
    assert len(valid_df) == n_rows - 1
    assert len(np.unique(valid_df)) == len(valid_df)


def test_drop_timesteps():
    """Test exclusion of specified timesteps and error on full drop."""
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
    df_out = drop_timesteps(df, timesteps)

    # Test np.NaT is conserved
    assert np.isnan(df_out["time"])[0]

    # Test all timesteps were dropped
    assert not np.any(df_out["time"].isin(timesteps))

    # Test error is raised if all timesteps are dropped
    with pytest.raises(ValueError):
        drop_timesteps(df, timesteps=time)


def test_drop_time_periods():
    """Test exclusion of specified time intervals and proper error flows."""
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
    df_out = drop_time_periods(df, time_periods=outside_time_period)
    pd.testing.assert_frame_equal(df_out, df)

    # Test inside time_periods
    df_out = drop_time_periods(df, time_periods=inside_time_periods)
    assert not np.any(df_out["time"].between(inside_time_periods[0][0], inside_time_periods[0][1], inclusive="both"))

    # Test raise error if all rows are discarded
    with pytest.raises(ValueError):
        drop_time_periods(df, time_periods=full_time_period)

    # Test code do not break if all rows are removed after first time_period iteration
    # --> Would raise IndexError otherwise
    time_periods = [full_time_period[0], [inside_time_periods[0]]]
    with pytest.raises(ValueError):
        drop_time_periods(df, time_periods=time_periods)


def create_fake_csv(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


# import pathlib
# tmp_path = pathlib.Path("/tmp/18")
# tmp_path.mkdir(parents=True)


def test_read_raw_text_file(tmp_path):
    """Test reading of raw text into DataFrame and handling of empty files."""
    # Create a valid test file
    filepath = os.path.join(tmp_path, "test.csv")
    data = {"att_1": ["11", "21"], "att_2": ["12", "22"]}
    create_fake_csv(filepath, data)

    reader_kwargs = {}
    reader_kwargs["delimiter"] = ","
    reader_kwargs["header"] = 0
    reader_kwargs["engine"] = "python"

    df = read_raw_text_file(
        filepath=filepath,
        column_names=["att_1", "att_2"],
        reader_kwargs=reader_kwargs,
    )
    df_expected = pd.DataFrame(data)
    assert df.equals(df_expected)

    # Test with an empty file without column
    filepath = os.path.join(tmp_path, "test_empty.csv")
    print(filepath)
    data = {}
    create_fake_csv(filepath, data)

    # Call the function and catch the exception
    with pytest.raises(ValueError, match="The following file is empty"):
        read_raw_text_file(
            filepath=filepath,
            column_names=[],
            reader_kwargs=reader_kwargs,
        )

    # Test with an empty file with column
    filepath = os.path.join(tmp_path, "test_empty2.csv")
    print(filepath)
    data = {"att_1": [], "att_2": []}
    create_fake_csv(filepath, data)

    # Call the function and catch the exception
    with pytest.raises(ValueError, match="The following file is empty"):
        read_raw_text_file(
            filepath=filepath,
            column_names=["att_1", "att_2"],
            reader_kwargs=reader_kwargs,
        )


def test_write_l0a(tmp_path):
    """Test writing and reading L0A parquet files and type validation."""
    # create dummy dataframe
    data = [{"a": "1", "b": "2", "c": "3"}, {"a": "2", "b": "2", "c": "3"}]
    df = pd.DataFrame(data).set_index("a")
    df["time"] = pd.Timestamp.now().to_numpy().astype("M8[ns]")  # open by default as [ns]. Now() returns as [us]

    # Write parquet file
    filepath = os.path.join(tmp_path, "fake_data_sample.parquet")
    write_l0a(df, filepath, force=True, verbose=False)

    # Read parquet file
    df_written = read_l0a_dataframe([filepath], verbose=False)

    # Check if parquet file are similar
    is_equal = df.equals(df_written)
    assert is_equal

    # Test error is raised when bad parquet file
    with pytest.raises(ValueError):
        write_l0a("dummy_object", filepath, force=True, verbose=False)


class TestReadRawTextFiles:
    def test_empty_filepaths_raises(self, tmp_path):
        """Should raise ValueError when no filepaths are provided."""
        # Define verbose argument
        verbose = False

        # Define sensor name
        sensor_name = "OTT_Parsivel"

        # Define a dummy reader
        def reader(filepath, logger=None):
            return pd.read_parquet(filepath)

        # Test raise value error if empty filepaths list is passed
        with pytest.raises(ValueError, match="'filepaths' must contains at least 1 filepath"):
            read_raw_text_files(
                filepaths=[],
                reader=reader,
                sensor_name=sensor_name,
                verbose=verbose,
            )

    def test_multiple_files_concatenated(self, tmp_path):
        """Should return concatenated DataFrame for multiple valid filepaths."""
        # Define verbose argument
        verbose = False

        # Define sensor name
        sensor_name = "OTT_Parsivel"

        # Create dummy DataFrames
        raw_str = ",".join(["000"] * 1024)
        df1 = pd.DataFrame(
            {
                "time": pd.date_range(start="2025-01-01", periods=3, freq="1min"),
                "raw_drop_number": [raw_str] * 3,
            },
        )
        df2 = pd.DataFrame(
            {
                "time": pd.date_range(start="2025-01-02", periods=3, freq="1min"),
                "raw_drop_number": [raw_str] * 3,
            },
        )

        # Define raw filepaths
        file1 = tmp_path / "test_file1.csv"
        file2 = tmp_path / "test_file2.csv"

        # Create raw files
        df1.to_parquet(str(file1))
        df2.to_parquet(str(file2))

        # Define a dummy reader
        def reader(filepath, logger=None):
            return pd.read_parquet(filepath)

        # Test the function returns the expected dataframe
        df_output = read_raw_text_files(
            filepaths=[file1, file2],
            reader=reader,
            sensor_name=sensor_name,
            verbose=verbose,
        )
        df_expected = pd.concat([df1, df2]).reset_index(drop=True)
        df_expected["time"] = df_expected["time"].astype("M8[ns]")
        pd.testing.assert_frame_equal(df_output, df_expected)

    def test_single_filepath_string(self, tmp_path):
        """Should accept a single filepath string and return its DataFrame."""
        # Define verbose argument
        verbose = False

        # Define sensor name
        sensor_name = "OTT_Parsivel"

        # Create dummy DataFrame
        raw_str = ",".join(["000"] * 1024)
        df1 = pd.DataFrame(
            {
                "time": pd.date_range(start="2025-01-01", periods=3, freq="1min"),
                "raw_drop_number": [raw_str] * 3,
            },
        )

        # Define raw filepath
        file1 = tmp_path / "test_file1.csv"

        # Create raw file
        df1.to_parquet(str(file1))

        # Define a dummy reader
        def reader(filepath, logger=None):
            return pd.read_parquet(filepath)

        # Test the function accept a single filepath (as string)
        df_output = read_raw_text_files(
            filepaths=str(file1),
            reader=reader,
            sensor_name=sensor_name,
            verbose=verbose,
        )
        df1["time"] = df1["time"].astype("M8[ns]")
        pd.testing.assert_frame_equal(df_output, df1)

    def test_skips_bad_filepaths(self, tmp_path):
        """Should skip invalid filepaths and concatenate only valid ones."""
        # Define verbose argument
        verbose = False

        # Define sensor name
        sensor_name = "OTT_Parsivel"

        # Create dummy DataFrames
        raw_str = ",".join(["000"] * 1024)
        df1 = pd.DataFrame(
            {
                "time": pd.date_range(start="2025-01-01", periods=3, freq="1min"),
                "raw_drop_number": [raw_str] * 3,
            },
        )
        df2 = pd.DataFrame(
            {
                "time": pd.date_range(start="2025-01-02", periods=3, freq="1min"),
                "raw_drop_number": [raw_str] * 3,
            },
        )

        # Define raw filepaths
        file1 = tmp_path / "test_file1.csv"
        file2 = tmp_path / "test_file2.csv"

        # Create raw files
        df1.to_parquet(str(file1))
        df2.to_parquet(str(file2))

        # Define a dummy reader
        def reader(filepath, logger=None):
            return pd.read_parquet(filepath)

        # Test bad filepath is skipped
        df_output = read_raw_text_files(
            filepaths=[file1, file2, "dummy_path"],
            reader=reader,
            sensor_name=sensor_name,
            verbose=verbose,
        )
        df_expected = pd.concat([df1, df2]).reset_index(drop=True)
        df_expected["time"] = df_expected["time"].astype("M8[ns]")
        pd.testing.assert_frame_equal(df_output, df_expected)

    def test_all_bad_filepaths_raise(self, tmp_path):
        """Should raise ValueError if no file is successfully read."""
        # Define verbose argument
        verbose = False

        # Define sensor name
        sensor_name = "OTT_Parsivel"

        # Define a dummy reader that always fails
        def reader(filepath, logger=None):
            raise ValueError("fail")

        # Test bad filepath is skipped
        with pytest.raises(ValueError, match="Any raw file could be read"):
            read_raw_text_files(
                filepaths=["dummy1", "dummy2"],
                reader=reader,
                sensor_name=sensor_name,
                verbose=verbose,
            )


class TestReadL0ADataFrame:
    def test_read_single_l0a_file(self, tmp_path):
        """Test reading a single L0A parquet file."""
        # Create dummy dataframe
        data = [{"a": "1", "b": "2", "c": "3"}, {"a": "2", "b": "2", "c": "3"}]
        df = pd.DataFrame(data)
        df["time"] = pd.Timestamp.now()

        # Save dataframe to parquet file
        filepath = os.path.join(tmp_path, "fake_data_sample.parquet")
        df.to_parquet(filepath, compression="gzip")

        # Read written parquet file
        df_written = read_l0a_dataframe(filepath, False)
        df["time"] = df["time"].astype("M8[ns]")
        pd.testing.assert_frame_equal(df_written, df)

    def test_read_multiple_l0a_files(self, tmp_path):
        """Test reading multiple L0A parquet files."""
        filepaths = []
        list_df = []
        for i in [0, 1]:
            # create dummy dataframe
            data = [{"a": "1", "b": "2", "c": "3"}, {"a": "2", "b": "2", "c": "3"}]
            df = pd.DataFrame(data).set_index("a")
            df["time"] = pd.Timestamp.now()

            # save dataframe to parquet file
            filepath = os.path.join(
                tmp_path,
                f"fake_data_sample_{i}.parquet",
            )
            df.to_parquet(filepath, compression="gzip")
            filepaths.append(filepath)
            list_df.append(df)

        # Create concatenate dataframe
        df_concatenated = pd.concat(list_df, axis=0, ignore_index=True)

        # Sort by increasing time
        df_concatenated = df_concatenated.sort_values(by="time")

        # Read written parquet files
        df_written = read_l0a_dataframe(filepaths, verbose=False)
        df_concatenated["time"] = df_concatenated["time"].astype("M8[ns]")
        pd.testing.assert_frame_equal(df_written, df_concatenated)

    def test_raise_type_filepaths(self, tmp_path):
        """Test raise error with bad filepaths type."""
        # Assert raise error if filepaths is not a list or string
        with pytest.raises(TypeError, match="Expecting filepaths to be a string or a list of strings."):
            read_l0a_dataframe(1, verbose=False)
