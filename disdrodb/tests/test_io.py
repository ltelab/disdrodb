import os
import datetime
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from disdrodb.L0 import io
import importlib.metadata


PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pytest_files"
)


PATH_FILE_WINDOWS = (
    "\\DISDRODB\\Raw\\institute_name\\campaign_name\\station_name\\file_name.tar.xz"
)
PATH_FILE_LINUX = (
    "/DISDRODB/Raw/institute_name/campaign_name/station_name/file_name.tar.xz"
)


@pytest.mark.parametrize("path_raw_data", [PATH_FILE_LINUX, PATH_FILE_WINDOWS])
def test_infer_institute_from_fpath(path_raw_data):
    assert io.infer_institute_from_fpath(path_raw_data) == "institute_name"


@pytest.mark.parametrize("path_raw_data", [PATH_FILE_LINUX, PATH_FILE_WINDOWS])
def test_infer_campaign_from_fpath(path_raw_data):
    assert io.infer_campaign_from_fpath(path_raw_data) == "campaign_name"


@pytest.mark.parametrize("path_raw_data", [PATH_FILE_LINUX, PATH_FILE_WINDOWS])
def test_infer_station_id_from_fpath(path_raw_data):
    assert io.infer_station_id_from_fpath(path_raw_data) == "station_name"


PATH_FILE_WINDOWS = "\\DISDRODB\\Raw\\institute_name\\campaign_name"
PATH_FILE_LINUX = "/DISDRODB/Raw/institute_name/campaign_name"


@pytest.mark.parametrize("path_raw_data", [PATH_FILE_LINUX, PATH_FILE_WINDOWS])
def test_get_campaign_name(path_raw_data):
    assert io.get_campaign_name(path_raw_data) == "CAMPAIGN_NAME"


@pytest.mark.parametrize("path_raw_data", [PATH_FILE_LINUX, PATH_FILE_WINDOWS])
def test_get_data_source(path_raw_data):
    assert io.get_data_source(path_raw_data) == "INSTITUTE_NAME"


def test_get_dataset_min_max_time():
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})
    res = io.get_dataset_min_max_time(df)
    assert all(pd.to_datetime(res, format="%Y-%m-%d") == [start_date, end_date])


PATH_PROCESS_DIR_WINDOWS = "\\DISDRODB\\Processed"
PATH_PROCESS_DIR_LINUX = "/DISDRODB/Processed"


@pytest.mark.parametrize(
    "path_process_dir", [PATH_PROCESS_DIR_WINDOWS, PATH_PROCESS_DIR_LINUX]
)
def test_get_L0A_dir(path_process_dir):
    res = (
        io.get_L0A_dir(path_process_dir, "station_id")
        .replace(path_process_dir, "")
        .replace("\\", "")
        .replace("/", "")
    )
    assert res == "L0Astation_id"


@pytest.mark.parametrize(
    "path_process_dir", [PATH_PROCESS_DIR_WINDOWS, PATH_PROCESS_DIR_LINUX]
)
def test_get_L0B_dir(path_process_dir):
    res = (
        io.get_L0B_dir(path_process_dir, "station_id")
        .replace(path_process_dir, "")
        .replace("\\", "")
        .replace("/", "")
    )
    assert res == "L0Bstation_id"


def test_get_L0A_fpath():
    """
    Test the naming and the path of the L0A file
    Note that this test needs "/pytest_files/test_folders_files_structure/DISDRODB/Processed/institute_name/campaign_name/metadata/STATIONID.yml"
    """

    # Set variables
    institute_name = "INSTITUTE_NAME"
    campaign_name = "CAMPAIGN_NAME"
    station_id = "STATIONID"
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
    start_date_str = start_date.strftime("%Y%m%d%H%M%S")
    end_date_str = end_date.strftime("%Y%m%d%H%M%S")

    # Set paths
    path_campaign_name = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
    )

    # Create dataframe
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})

    # Set version (based on the version included into the setup.py to create the pypi package)
    version = importlib.metadata.version("disdrodb").replace(".", "-")
    if version == "-VERSION-PLACEHOLDER-":
        version = "dev"

    # Test the function
    res = io.get_L0A_fpath(df, path_campaign_name, station_id)

    # Define expected results
    expected_name = f"DISDRODB.L0A.{campaign_name.upper()}.{station_id}.s{start_date_str}.e{end_date_str}.{version}.parquet"
    expected_path = os.path.join(path_campaign_name, "L0A", station_id, expected_name)
    assert res == expected_path


def test_get_L0B_fpath():
    """
    Test the naming and the path of the L0B file
    Note that this test needs "/pytest_files/test_folders_files_structure/DISDRODB/Processed/institute_name/campaign_name/metadata/STATIONID.yml"
    """

    # Set variables
    institute_name = "INSTITUTE_NAME"
    campaign_name = "CAMPAIGN_NAME"
    station_id = "STATIONID"
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
    start_date_str = start_date.strftime("%Y%m%d%H%M%S")
    end_date_str = end_date.strftime("%Y%m%d%H%M%S")

    # Set paths
    path_campaign_name = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
    )

    # Create xarray object
    timesteps = pd.date_range(start=start_date, end=end_date)
    data = np.zeros(timesteps.shape)
    ds = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": pd.date_range(start=start_date, end=end_date)},
    )

    # Set version (based on the version included into the setup.py to create the pypi package)
    version = importlib.metadata.version("disdrodb").replace(".", "-")
    if version == "-VERSION-PLACEHOLDER-":
        version = "dev"

    # Test the function
    res = io.get_L0B_fpath(ds, path_campaign_name, station_id)

    # Define expected results
    expected_name = f"DISDRODB.L0B.{campaign_name.upper()}.{station_id}.s{start_date_str}.e{end_date_str}.{version}.nc"
    expected_path = os.path.join(path_campaign_name, "L0B", station_id, expected_name)
    assert res == expected_path


folder_name = "folder_creation_deletion_test"
path_file_temp = os.path.join(
    PATH_TEST_FOLDERS_FILES, "test_folders_files_creation", folder_name
)


def test__create_directory():
    io._create_directory(path_file_temp)
    if os.path.exists(path_file_temp):
        res = True
    else:
        res = False
    assert res


def test__remove_directory():

    # Create empty folder if not exists
    if not os.path.exists(path_file_temp):
        io._create_directory(path_file_temp)

    # Remove this folder
    io._remove_if_exists(path_file_temp, True)

    # Test the removal
    if os.path.exists(path_file_temp):
        res = False
    else:
        res = True
    assert res


def test_parse_fpath():
    path_dir_windows_in = "\\DISDRODB\\Processed\\institute_name\\campaign_name\\"
    path_dir_windows_out = "\\DISDRODB\\Processed\\institute_name\\campaign_name"
    assert io.parse_fpath(path_dir_windows_in) == path_dir_windows_out

    path_dir_linux_in = "/DISDRODB/Processed/institute_name/campaign_name/"
    path_dir_linux_out = "/DISDRODB/Processed/institute_name/campaign_name"
    assert io.parse_fpath(path_dir_linux_in) == path_dir_linux_out


def test_check_raw_dir():
    # Set variables
    institute_name = "INSTITUTE_NAME"
    campaign_name = "CAMPAIGN_NAME"

    # Set paths
    path_campaign_name = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Raw",
        institute_name,
        campaign_name,
    )

    assert io.check_raw_dir(path_campaign_name) is None


def test_check_processed_dir():
    """
    This function will create the processed folder structure under test_folders_files_creation folder.
    """
    # Set variables
    institute_name = "INSTITUTE_NAME"
    campaign_name = "CAMPAIGN_NAME"

    # Set paths
    path_campaign_name = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
    )

    assert io.check_processed_dir(path_campaign_name, force=True) is None


def test_check_campaign_name():
    campaign_name = "CAMPAIGN_NAME"
    institute_name = "INSTITUTE_NAME"
    path_raw = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Raw",
        institute_name,
        campaign_name,
    )
    path_process = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
    )

    assert io.check_campaign_name(path_raw, path_process) == campaign_name


def test_check_directories():
    campaign_name = "CAMPAIGN_NAME"
    institute_name = "INSTITUTE_NAME"
    path_raw = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Raw",
        institute_name,
        campaign_name,
    )
    path_process = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
    )

    assert io.check_directories(path_raw, path_process, force=True) == (
        path_raw,
        path_process,
    )


def test_check_metadata_dir():
    campaign_name = "CAMPAIGN_NAME"
    institute_name = "INSTITUTE_NAME"
    path_process = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
    )

    assert io.check_metadata_dir(path_process) is None


def test_copy_metadata_from_raw_dir():
    campaign_name = "CAMPAIGN_NAME"
    institute_name = "INSTITUTE_NAME"
    path_raw = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Raw",
        institute_name,
        campaign_name,
    )
    path_process = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
    )

    # create processed structure if needed
    if not os.path.exists(path_process):
        os.makedirs(path_process)

    path_metadata = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
        "metadata",
    )
    if not os.path.exists(path_metadata):
        os.makedirs(path_metadata)

    assert io.copy_metadata_from_raw_dir(path_raw, path_process) is None


def test_create_directory_structure():
    campaign_name = "CAMPAIGN_NAME"
    institute_name = "INSTITUTE_NAME"
    path_raw = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Raw",
        institute_name,
        campaign_name,
    )
    path_process = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "DISDRODB",
        "Processed",
        institute_name,
        campaign_name,
    )

    assert io.create_directory_structure(path_raw, path_process) is None


def test__read_L0A():
    # create dummy dataframe
    data = [{"a": "1", "b": "2"}, {"a": "2", "b": "2", "c": "3"}]
    df = pd.DataFrame(data)

    # save dataframe to parquet file
    path_parquet_file = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "fake_data_sample.parquet",
    )
    df.to_parquet(path_parquet_file, compression="gzip")

    # read written parquet file
    df_written = io._read_L0A(path_parquet_file, False)

    assert df.equals(df_written)


def test_read_L0A_dataframe():

    list_of_parquet_file_paths = list()

    for i in [0, 1]:
        # create dummy dataframe
        data = [{"a": "1", "b": "2", "c": "3"}, {"a": "2", "b": "2", "c": "3"}]
        df = pd.DataFrame(data).set_index("a")
        df["time"] = pd.Timestamp.now()

        # save dataframe to parquet file
        path_parquet_file = os.path.join(
            PATH_TEST_FOLDERS_FILES,
            "test_folders_files_creation",
            f"fake_data_sample_{i}.parquet",
        )
        df.to_parquet(path_parquet_file, compression="gzip")
        list_of_parquet_file_paths.append(path_parquet_file)

        # create concatenate dataframe
        if i == 0:
            df_concatenate = df
        else:
            df_concatenate = pd.concat([df, df_concatenate], axis=0, ignore_index=True)

    # Drop duplicated values
    df_concatenate = df_concatenate.drop_duplicates(subset="time")
    # Sort by increasing time
    df_concatenate = df_concatenate.sort_values(by="time")

    # read written parquet files
    df_written = io.read_L0A_dataframe(list_of_parquet_file_paths, False)

    # Create lists
    df_concatenate_list = df_concatenate.values.tolist()
    df_written_list = df_written.values.tolist()

    # Compare lists
    comparaison = df_written_list == df_concatenate_list

    assert comparaison
