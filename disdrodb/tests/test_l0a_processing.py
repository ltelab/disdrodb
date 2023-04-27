import os
import pytest
import numpy as np
import pandas as pd
from disdrodb.l0 import l0a_processing
from disdrodb.l0 import io


from disdrodb.l0.l0a_processing import (
    _check_df_sanitizer_fun,
    _check_not_empty_dataframe,
    _check_matching_column_number,
    remove_issue_timesteps,
    cast_column_dtypes,
)

PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "pytest_files",
)


def test_remove_issue_timesteps():
    # Create dummy dataframe
    df = pd.DataFrame({"time": [1, 2, 3, 4, 5], "col1": [0, 1, 2, 3, 4]})

    # Create dummy issue dictionary with timesteps to remove
    issue_dict = {"timesteps": [2, 4]}

    # Call function to remove problematic timesteps
    df_cleaned = remove_issue_timesteps(df, issue_dict)

    # Check that problematic timesteps were removed
    assert set(df_cleaned["time"]) == {1, 3, 5}


def test_preprocess_reader_kwargs():
    # Test that the function removes the 'dtype' key from the reader_kwargs dict
    reader_kwargs = {"dtype": "int64", "other_key": "other_value", "delimiter": ","}
    preprocessed_kwargs = l0a_processing.preprocess_reader_kwargs(reader_kwargs)
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
    preprocessed_kwargs = l0a_processing.preprocess_reader_kwargs(
        reader_kwargs,
    )
    assert "blocksize" not in preprocessed_kwargs
    assert "assume_missing" not in preprocessed_kwargs
    assert "other_key" in preprocessed_kwargs

    # Test raise error if delimiter is not specified
    reader_kwargs = {"dtype": "int64", "other_key": "other_value"}
    with pytest.raises(ValueError):
        l0a_processing.preprocess_reader_kwargs(reader_kwargs)


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
    # Test valid string (convertable to numeric, after split by ,)
    s = "000,001,000"
    assert l0a_processing._is_not_corrupted(s)
    # Test corrupted string (not convertable to numeric, after split by ,)
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
    df_out = cast_column_dtypes(df, "OTT_Parsivel", verbose=False)
    # Check that the output dataframe has the correct column types
    assert df_out["time"].dtype == "datetime64[s]"
    assert df_out["station_number"].dtype == "object"
    assert df_out["altitude"].dtype == "float64"


def test_coerce_corrupted_values_to_nan():
    # not tested yet because relies on config files that can be modified
    # function_return = l0a_processing.coerce_corrupted_values_to_nan()
    assert 1 == 1


def test_strip_string_spaces():
    # not tested yet because relies on config files that can be modified
    # function_return = l0a_processing.strip_string_spaces()
    assert 1 == 1


def test_remove_rows_with_missing_time():
    # Create dataframe
    n_rows = 3
    time = pd.date_range(start="2023-01-01", periods=n_rows, freq="D")
    df = pd.DataFrame({"time": time})

    # Add Nat value to a single rows of the time column
    df.at[0, "time"] = np.datetime64("NaT")
    # Test it remove the unvalid timestep
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


def test_read_raw_file_list():
    # not tested yet because relies on config files that can be modified
    # function_return = l0a_processing.read_raw_file_list()
    assert 1 == 1


def test_write_l0a():
    # create dummy dataframe
    data = [{"a": "1", "b": "2", "c": "3"}, {"a": "2", "b": "2", "c": "3"}]
    df = pd.DataFrame(data).set_index("a")
    df["time"] = pd.Timestamp.now()

    # Write parquet file
    path_parquet_file = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "fake_data_sample.parquet",
    )
    l0a_processing.write_l0a(df, path_parquet_file, True, False)

    # Read parquet file
    df_written = io.read_L0A_dataframe([path_parquet_file], False)

    # Check if parquet file are similar
    is_equal = df.equals(df_written)

    assert is_equal
