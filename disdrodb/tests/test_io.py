import os
import datetime
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from disdrodb.l0 import io


PATH_TEST_FOLDERS_FILES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytest_files")


def get_disdrodb_path():
    # Assert retrieve correct disdrodb path
    disdrodb_path = os.path.join("DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    assert io.get_disdrodb_path(path) == disdrodb_path

    # Assert raise error if not disdrodb path
    disdrodb_path = os.path.join("no_disdro_dir", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    with pytest.raises(ValueError):
        io.get_disdrodb_path(path)

    # Assert raise error if not valid DISDRODB directory
    disdrodb_path = os.path.join("DISDRODB_UNVALID", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    with pytest.raises(ValueError):
        io.get_disdrodb_path(path)

    # Assert it takes the right most DISDRODB occurence
    disdrodb_path = os.path.join("DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_occurence", "DISDRODB", "DISDRODB", "directory", disdrodb_path)
    assert io.get_disdrodb_path(path) == disdrodb_path

    # Assert behaviour when path == disdrodb_dir
    disdrodb_dir = os.path.join("home", "DISDRODB")
    io.get_disdrodb_path(disdrodb_dir) == "DISDRODB"


def get_disdrodb_dir():
    # Assert retrieve correct disdrodb path
    disdrodb_dir = os.path.join("whatever_path", "is", "before", "DISDRODB")
    disdrodb_path = os.path.join("Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join(disdrodb_dir, disdrodb_path)
    assert get_disdrodb_dir(path) == disdrodb_dir

    # Assert raise error if not disdrodb path
    disdrodb_dir = os.path.join("whatever_path", "is", "before", "NO_DISDRODB")
    disdrodb_path = os.path.join("Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join(disdrodb_dir, disdrodb_path)
    with pytest.raises(ValueError):
        io.get_disdrodb_dir(path)

    # Assert behaviour when path == disdrodb_dir
    disdrodb_dir = os.path.join("home", "DISDRODB")
    io.get_disdrodb_dir(disdrodb_dir) == disdrodb_dir


def _get_disdrodb_path_components():
    # Assert retrieve correct disdrodb path
    path_components = ["DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME"]
    disdrodb_path = os.path.join(*path_components)
    path = os.path.join("whatever_path", disdrodb_path)
    assert _get_disdrodb_path_components(path) == path_components


def test_get_data_source():
    # Assert retrieve correct
    path = os.path.join("whatever_path", "DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    assert io.get_data_source(path) == "DATA_SOURCE"

    # Assert raise error if path stop at Raw or Processed
    path = os.path.join("whatever_path", "DISDRODB", "Raw")
    with pytest.raises(ValueError):
        io.get_data_source(path)

    # Assert raise error if path not within DISDRODB
    path = os.path.join("whatever_path", "is", "not", "valid")
    with pytest.raises(ValueError):
        io.get_data_source(path)


def test_get_campaign_name():
    # Assert retrieve correct
    path = os.path.join("whatever_path", "DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME", "...")
    assert io.get_campaign_name(path) == "CAMPAIGN_NAME"

    # Assert raise error if path stop at Raw or Processed
    path = os.path.join("whatever_path", "DISDRODB", "Raw")
    with pytest.raises(ValueError):
        io.get_campaign_name(path)

    # Assert raise error if path not within DISDRODB
    path = os.path.join("whatever_path", "is", "not", "valid")
    with pytest.raises(ValueError):
        io.get_campaign_name(path)


####--------------------------------------------------------------------------.

PATH_PROCESS_DIR_WINDOWS = "\\DISDRODB\\Processed"
PATH_PROCESS_DIR_LINUX = "/DISDRODB/Processed"


def test_get_dataset_min_max_time():
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})
    res = io.get_dataset_min_max_time(df)
    assert all(pd.to_datetime(res, format="%Y-%m-%d") == [start_date, end_date])


@pytest.mark.parametrize("path_process_dir", [PATH_PROCESS_DIR_WINDOWS, PATH_PROCESS_DIR_LINUX])
def test_get_L0A_dir(path_process_dir):
    res = (
        io.get_L0A_dir(path_process_dir, "STATION_NAME")
        .replace(path_process_dir, "")
        .replace("\\", "")
        .replace("/", "")
    )
    assert res == "L0ASTATION_NAME"


@pytest.mark.parametrize("path_process_dir", [PATH_PROCESS_DIR_WINDOWS, PATH_PROCESS_DIR_LINUX])
def test_get_L0B_dir(path_process_dir):
    res = (
        io.get_L0B_dir(path_process_dir, "STATION_NAME")
        .replace(path_process_dir, "")
        .replace("\\", "")
        .replace("/", "")
    )
    assert res == "L0BSTATION_NAME"


def test_get_L0A_fpath():
    """
    Test the naming and the path of the L0A file
    Note that this test needs "/pytest_files/test_folders_files_structure/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME/
    metadata/STATION_NAME.yml"
    """
    from disdrodb.l0.standards import PRODUCT_VERSION

    # Set variables
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
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
        data_source,
        campaign_name,
    )

    # Create dataframe
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})

    # Test the function
    res = io.get_L0A_fpath(df, path_campaign_name, station_name)

    # Define expected results
    expected_name = (
        f"L0A.{campaign_name.upper()}.{station_name}.s{start_date_str}.e{end_date_str}.{PRODUCT_VERSION}.parquet"
    )
    expected_path = os.path.join(path_campaign_name, "L0A", station_name, expected_name)
    assert res == expected_path


def test_get_L0B_fpath():
    """
    Test the naming and the path of the L0B file
    Note that this test needs "/pytest_files/test_folders_files_structure/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME/
    metadata/STATION_NAME.yml"
    """
    from disdrodb.l0.standards import PRODUCT_VERSION

    # Set variables
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
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
        data_source,
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

    # Test the function
    res = io.get_L0B_fpath(ds, path_campaign_name, station_name)

    # Define expected results
    expected_name = f"L0B.{campaign_name.upper()}.{station_name}.s{start_date_str}.e{end_date_str}.{PRODUCT_VERSION}.nc"
    expected_path = os.path.join(path_campaign_name, "L0B", station_name, expected_name)
    assert res == expected_path


####--------------------------------------------------------------------------.


def test_check_glob_pattern():
    function_return = io.check_glob_pattern("1")
    assert function_return is None

    with pytest.raises(TypeError, match="Expect pattern as a string."):
        io.check_glob_pattern(1)

    with pytest.raises(ValueError, match="glob_pattern should not start with /"):
        io.check_glob_pattern("/1")


def test_get_raw_file_list():
    path_test_directory = os.path.join(PATH_TEST_FOLDERS_FILES, "test_l0a_processing", "files")

    station_name = "STATION_NAME"

    # Test that the function returns the correct number of files in debugging mode
    file_list = io.get_raw_file_list(
        raw_dir=path_test_directory,
        station_name=station_name,
        glob_patterns="*.txt",
        debugging_mode=True,
    )
    assert len(file_list) == 2  # max(2, 3)

    # Test that the function returns the correct number of files in normal mode
    file_list = io.get_raw_file_list(raw_dir=path_test_directory, station_name=station_name, glob_patterns="*.txt")
    assert len(file_list) == 2

    # Test that the function raises an error if the glob_patterns is not a str or list
    with pytest.raises(ValueError, match="'glob_patterns' must be a str or list of strings."):
        io.get_raw_file_list(raw_dir=path_test_directory, station_name=station_name, glob_patterns=1)

    # Test that the function raises an error if no files are found
    with pytest.raises(ValueError):
        io.get_raw_file_list(
            raw_dir=path_test_directory,
            station_name=station_name,
            glob_patterns="*.csv",
        )


####--------------------------------------------------------------------------.

folder_name = "folder_creation_deletion_test"
path_file_temp = os.path.join(PATH_TEST_FOLDERS_FILES, "test_folders_files_creation", folder_name)


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
    path_dir_windows_in = "\\DISDRODB\\Processed\\DATA_SOURCE\\CAMPAIGN_NAME\\"
    path_dir_windows_out = "\\DISDRODB\\Processed\\DATA_SOURCE\\CAMPAIGN_NAME"
    assert io._parse_fpath(path_dir_windows_in) == path_dir_windows_out

    path_dir_linux_in = "/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME/"
    path_dir_linux_out = "/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME"
    assert io._parse_fpath(path_dir_linux_in) == path_dir_linux_out


def test_check_raw_dir():
    # Set variables
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"

    # Set paths
    raw_dir = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )

    assert io.check_raw_dir(raw_dir) == raw_dir


def test_check_campaign_name():
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    path_raw = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )
    path_process = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
    )

    assert io._check_campaign_name(path_raw, path_process) == campaign_name


def test_copy_station_metadata():
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "STATION_NAME"
    raw_dir = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )
    processed_dir = os.path.join(
        PATH_TEST_FOLDERS_FILES,
        "test_folders_files_creation",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
    )
    destination_metadata_dir = os.path.join(processed_dir, "metadata")

    # Ensure processed_dir and metadata folder exists
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if not os.path.exists(destination_metadata_dir):
        os.makedirs(destination_metadata_dir)

    # Define expected metadata file name
    expected_metadata_fpath = os.path.join(destination_metadata_dir, f"{station_name}.yml")
    # Ensure metadata file does not exist
    if os.path.exists(expected_metadata_fpath):
        os.remove(expected_metadata_fpath)
    assert not os.path.exists(expected_metadata_fpath)

    # Check the function returns None
    assert io._copy_station_metadata(raw_dir, processed_dir, station_name) is None

    # Check the function has copied the file
    assert os.path.exists(expected_metadata_fpath)


# def test_create_initial_directory_structure():
#     campaign_name = "CAMPAIGN_NAME"
#     data_source = "DATA_SOURCE"
#     station_name = "STATION_NAME"
#     product_level = "L0A"
#     force = True
#     verbose=False

#     raw_dir = os.path.join(
#         PATH_TEST_FOLDERS_FILES,
#         "test_folders_files_structure",
#         "DISDRODB",
#         "Raw",
#         data_source,
#         campaign_name,
#     )
#     processed_dir = os.path.join(
#         PATH_TEST_FOLDERS_FILES,
#         "test_folders_files_creation",
#         "DISDRODB",
#         "Processed",
#         data_source,
#         campaign_name,
#     )
#     # Define expected directory
#     expected_product_dir = os.path.join(processed_dir, product_level)

#     # TODO:
#     # - Need to remove file to check function works, but then next test is invalidated
#     # - I think we need to create a default directory that we can reinitialize at each test !

#     # Remove directory if exists already
#     if os.path.exists(expected_product_dir):
#         shutil.rmtree(expected_product_dir)
#     assert not os.path.exists(expected_product_dir)

#     # Create directories
#     assert io.create_directory_structure(processed_dir=processed_dir,
#                                          product_level=product_level,
#                                          station_name=station_name,
#                                          force=force,
#                                          verbose=verbose,
#                                          ) is None
#     # Check the directory has been created
#     assert not os.path.exists(expected_product_dir)
#     # TODO:
#     # - check that if data are already present and force=False, raise Error


# def test_create_directory_structure():
#     campaign_name = "CAMPAIGN_NAME"
#     data_source = "DATA_SOURCE"
#     station_name = "STATION_NAME"
#     product_level = "L0B"
#     force = True
#     verbose=False

#     processed_dir = os.path.join(
#         PATH_TEST_FOLDERS_FILES,
#         "test_folders_files_creation",
#         "DISDRODB",
#         "Processed",
#         data_source,
#         campaign_name,
#     )
#     # Define expected directory
#     expected_product_dir = os.path.join(processed_dir, product_level)

#     # Remove directory if exists already
#     if os.path.exists(expected_product_dir):
#         shutil.rmtree(expected_product_dir)
#     assert not os.path.exists(expected_product_dir)

#     # Create directories
#     assert io.create_directory_structure(processed_dir=processed_dir,
#                                          product_level=product_level,
#                                          station_name=station_name,
#                                          force=force,
#                                          verbose=verbose,
#                                          ) is None
#     # Check the directory has been created
#     assert not os.path.exists(expected_product_dir)
#     # TODO - check that if data are already present and force=False, raise Error


####--------------------------------------------------------------------------.


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
