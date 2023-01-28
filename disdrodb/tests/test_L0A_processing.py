import os
import pytest
import pandas as pd
import dask.dataframe as dd
from disdrodb.L0 import L0A_processing
from disdrodb.L0 import io

PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "pytest_files",
)


def test_preprocess_reader_kwargs():
    # Test that the function removes the 'dtype' key from the reader_kwargs dict
    reader_kwargs = {"dtype": "int64", "other_key": "other_value"}
    preprocessed_kwargs = L0A_processing.preprocess_reader_kwargs(reader_kwargs)
    assert "dtype" not in preprocessed_kwargs
    assert "other_key" in preprocessed_kwargs

    # Test that the function removes the 'index_col' key when lazy=True
    reader_kwargs = {"index_col": 0, "other_key": "other_value"}
    preprocessed_kwargs = L0A_processing.preprocess_reader_kwargs(
        reader_kwargs, lazy=True
    )
    assert "index_col" not in preprocessed_kwargs
    assert "other_key" in preprocessed_kwargs

    # Test that the function removes the 'blocksize' and 'assume_missing' keys when lazy=False
    reader_kwargs = {
        "blocksize": 128,
        "assume_missing": True,
        "other_key": "other_value",
    }
    preprocessed_kwargs = L0A_processing.preprocess_reader_kwargs(
        reader_kwargs, lazy=False
    )
    assert "blocksize" not in preprocessed_kwargs
    assert "assume_missing" not in preprocessed_kwargs
    assert "other_key" in preprocessed_kwargs

    # Test that the function removes the 'zipped' and 'blocksize' keys when 'zipped' is True
    reader_kwargs = {
        "zipped": True,
        "blocksize": 128,
        "other_key": "other_value",
    }
    preprocessed_kwargs = L0A_processing.preprocess_reader_kwargs(reader_kwargs)
    assert "zipped" not in preprocessed_kwargs
    assert "blocksize" not in preprocessed_kwargs
    assert "other_key" in preprocessed_kwargs


def test_concatenate_dataframe():
    # Test that the function returns a Pandas dataframe
    df1 = pd.DataFrame({"time": [1, 2, 3], "value": [4, 5, 6]})
    df2 = pd.DataFrame({"time": [7, 8, 9], "value": [10, 11, 12]})
    concatenated_df = L0A_processing.concatenate_dataframe([df1, df2])
    assert isinstance(concatenated_df, pd.DataFrame)

    # Test that the function raises a ValueError if the list_df is empty
    with pytest.raises(ValueError, match="No objects to concatenate"):
        L0A_processing.concatenate_dataframe([])

    with pytest.raises(ValueError):
        L0A_processing.concatenate_dataframe(["not a dataframe"])
    
    with pytest.raises(ValueError):
        L0A_processing.concatenate_dataframe(["not a dataframe", "not a dataframe"])

def test_cast_column_dtypes():
    # not tested yet because relies on config files that can be modified

    assert 1 == 1


def test_coerce_corrupted_values_to_nan():
    # not tested yet because relies on config files that can be modified
    # function_return = L0A_processing.coerce_corrupted_values_to_nan()
    assert 1 == 1


def test_strip_string_spaces():
    # not tested yet because relies on config files that can be modified
    # function_return = L0A_processing.strip_string_spaces()
    assert 1 == 1


def test_read_raw_data():
    # this test relies on "\tests\pytest_files\test_L0A_processing\test_read_raw_data\data.csv"

    path_test_data = os.path.join(
        PATH_TEST_FOLDERS_FILES, "test_L0A_processing", "test_read_raw_data", "data.csv"
    )

    reader_kwargs = {}
    reader_kwargs["delimiter"] = ","
    reader_kwargs["header"] = 0
    reader_kwargs["engine"] = "python"

    r = L0A_processing.read_raw_data(
        filepath=path_test_data,
        column_names=["att_1", "att_2"],
        reader_kwargs=reader_kwargs,
    )

    assert r.to_dict() == {"att_1": {0: "11", 1: "21"}, "att_2": {0: "12", 1: "22"}}


def test_read_raw_data_zipped():
    # this test relies on "\tests\pytest_files\test_L0A_processing\test_read_raw_data\data.tar"

    path_test_data = os.path.join(
        PATH_TEST_FOLDERS_FILES, "test_L0A_processing", "test_read_raw_data", "data.tar"
    )

    reader_kwargs = {}
    reader_kwargs["delimiter"] = ","
    reader_kwargs["header"] = 0
    reader_kwargs["engine"] = "python"
    reader_kwargs["zipped"] = True

    r = L0A_processing._read_raw_data_zipped(
        path_test_data, column_names=["att_1", "att_2"], reader_kwargs=reader_kwargs,
    )

    assert r.to_dict() == {"att_1": {0: "11", 1: "21"}, "att_2": {0: "12", 1: "22"}}


def test_read_raw_file_list():
    # not tested yet because relies on config files that can be modified
    # function_return = L0A_processing.read_raw_file_list()
    assert 1 == 1


def test__write_to_parquet():
    # tested bellow
    # function_return = L0A_processing._write_to_parquet()
    assert 1 == 1


def test_write_df_to_parquet():

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
    L0A_processing.write_df_to_parquet(df, path_parquet_file, True, False)

    # Read parquet file
    df_written = io.read_L0A_dataframe([path_parquet_file], False)

    # Check if parquet file are similar
    is_equal = df.equals(df_written)

    assert is_equal
