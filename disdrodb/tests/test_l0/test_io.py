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
"""Test DISDRODB L0 Input/Output routines."""

import datetime
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb import __root_path__
from disdrodb.l0 import io
from disdrodb.tests.conftest import create_fake_metadata_file

TEST_DATA_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data")


def test_create_initial_directory_structure(tmp_path, mocker):
    force = False
    product = "LOA"

    # Define station info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"

    # Define Raw campaign directory structure
    raw_dir = tmp_path / "DISDRODB" / "Raw" / "DATA_SOURCE" / campaign_name
    raw_station_dir = raw_dir / "data" / station_name
    raw_station_dir.mkdir(parents=True)

    # - Add metadata
    metadata_dict = {}
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # - Add fake file
    fake_csv_file_path = os.path.join(raw_station_dir, f"{station_name}.csv")
    with open(fake_csv_file_path, "w") as f:
        f.write("fake csv file")

    # Define Processed campaign directory
    processed_dir = tmp_path / "DISDRODB" / "Processed" / data_source / campaign_name
    processed_dir.mkdir(parents=True)

    # Mock to pass metadata checks
    mocker.patch("disdrodb.metadata.check_metadata.check_metadata_compliance", return_value=None)

    # Execute create_initial_directory_structure
    io.create_initial_directory_structure(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        station_name=station_name,
        force=force,
        product=product,
    )

    # Test product directory has been created
    expected_folder_path = os.path.join(processed_dir, product)
    assert os.path.exists(expected_folder_path)


# def test_create_initial_directory_structure():
#     campaign_name = "CAMPAIGN_NAME"
#     data_source = "DATA_SOURCE"
#     station_name = "STATION_NAME"
#     product = "L0A"
#     force = True
#     verbose=False

#     raw_dir = os.path.join(
#         TEST_DATA_DIR,
#         "test_dir_structure",
#         "DISDRODB",
#         "Raw",
#         data_source,
#         campaign_name,
#     )
#     processed_dir = os.path.join(
#         TEST_DATA_DIR,
#         "test_dir_creation",
#         "DISDRODB",
#         "Processed",
#         data_source,
#         campaign_name,
#     )
#     # Define expected directory
#     expected_product_dir = os.path.join(processed_dir, product)

#     # TODO:
#     # - Need to remove file to check function works, but then next test is invalidated
#     # - I think we need to create a default directory that we can reinitialize at each test !

#     # Remove directory if exists already
#     if os.path.exists(expected_product_dir):
#         shutil.rmtree(expected_product_dir)
#     assert not os.path.exists(expected_product_dir)

#     # Create directories
#     assert io.create_directory_structure(processed_dir=processed_dir,
#                                          product=product,
#                                          station_name=station_name,
#                                          force=force,
#                                          verbose=verbose,
#                                          ) is None
#     # Check the directory has been created
#     assert not os.path.exists(expected_product_dir)
#     # TODO:
#     # - check that if data are already present and force=False, raise Error


def test_create_directory_structure(tmp_path, mocker):
    # from pathlib import Path
    # tmp_path = Path("/tmp/test12")
    # tmp_path.mkdir()

    force = False
    product = "L0B"

    # Define station info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"

    # Define Raw campaign directory structure
    raw_dir = tmp_path / "DISDRODB" / "Raw" / "DATA_SOURCE" / campaign_name
    raw_station_dir = raw_dir / "data" / station_name
    raw_station_dir.mkdir(parents=True)

    # - Add metadata
    metadata_dict = {}
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # - Add fake file
    fake_csv_file_path = os.path.join(raw_station_dir, f"{station_name}.csv")
    with open(fake_csv_file_path, "w") as f:
        f.write("fake csv file")

    # Define Processed campaign directory
    processed_dir = tmp_path / "DISDRODB" / "Processed" / data_source / campaign_name
    processed_dir.mkdir(parents=True)

    # subfolder_path = tmp_path / "DISDRODB" / "Processed" / campaign_name / "L0B"
    # subfolder_path.mkdir(parents=True)

    # Mock to pass some internal checks
    mocker.patch("disdrodb.api.io._get_list_stations_with_data", return_value=[station_name])
    mocker.patch("disdrodb.l0.io._check_pre_existing_station_data", return_value=None)

    # Execute create_directory_structure
    io.create_directory_structure(
        processed_dir=str(processed_dir), product=product, station_name=station_name, force=force, verbose=False
    )

    # Test product directory has been created
    l0a_folder_path = os.path.join(processed_dir, product)
    assert os.path.exists(l0a_folder_path)


# def testcreate_directory_structure():
#     campaign_name = "CAMPAIGN_NAME"
#     data_source = "DATA_SOURCE"
#     station_name = "STATION_NAME"
#     product = "L0B"
#     force = True
#     verbose=False

#     processed_dir = os.path.join(
#         TEST_DATA_DIR,
#         "test_dir_creation",
#         "DISDRODB",
#         "Processed",
#         data_source,
#         campaign_name,
#     )
#     # Define expected directory
#     expected_product_dir = os.path.join(processed_dir, product)

#     # Remove directory if exists already
#     if os.path.exists(expected_product_dir):
#         shutil.rmtree(expected_product_dir)
#     assert not os.path.exists(expected_product_dir)

#     # Create directories
#     assert io.create_directory_structure(processed_dir=processed_dir,
#                                          product=product,
#                                          station_name=station_name,
#                                          force=force,
#                                          verbose=verbose,
#                                          ) is None
#     # Check the directory has been created
#     assert not os.path.exists(expected_product_dir)
#     # TODO - check that if data are already present and force=False, raise Error


def test_check_campaign_name_consistency():
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    path_raw = os.path.join(
        TEST_DATA_DIR,
        "test_dir_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )
    path_process = os.path.join(
        TEST_DATA_DIR,
        "test_dir_creation",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
    )

    assert io._check_campaign_name_consistency(path_raw, path_process) == campaign_name


def test_copy_station_metadata():
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "STATION_NAME"
    raw_dir = os.path.join(
        TEST_DATA_DIR,
        "test_dir_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )
    processed_dir = os.path.join(
        TEST_DATA_DIR,
        "test_dir_creation",
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


####--------------------------------------------------------------------------.

PATH_PROCESS_DIR_WINDOWS = "\\DISDRODB\\Processed"
PATH_PROCESS_DIR_LINUX = "/DISDRODB/Processed"


def test__get_dataset_min_max_time():
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})
    res = io._get_dataset_min_max_time(df)
    assert all(pd.to_datetime(res, format="%Y-%m-%d") == [start_date, end_date])


@pytest.mark.parametrize("path_process_dir", [PATH_PROCESS_DIR_WINDOWS, PATH_PROCESS_DIR_LINUX])
def test_get_l0a_dir(path_process_dir):
    res = (
        io.get_l0a_dir(path_process_dir, "STATION_NAME")
        .replace(path_process_dir, "")
        .replace("\\", "")
        .replace("/", "")
    )
    assert res == "L0ASTATION_NAME"


@pytest.mark.parametrize("path_process_dir", [PATH_PROCESS_DIR_WINDOWS, PATH_PROCESS_DIR_LINUX])
def test_get_l0b_dir(path_process_dir):
    res = (
        io.get_l0b_dir(path_process_dir, "STATION_NAME")
        .replace(path_process_dir, "")
        .replace("\\", "")
        .replace("/", "")
    )
    assert res == "L0BSTATION_NAME"


def test_get_l0a_fpath():
    """
    Test the naming and the path of the L0A file
    Note that this test needs "/data/test_dir_structure/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME/
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
        TEST_DATA_DIR,
        "test_dir_structure",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
    )

    # Create dataframe
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})

    # Test the function
    res = io.get_l0a_fpath(df, path_campaign_name, station_name)

    # Define expected results
    expected_name = (
        f"L0A.{campaign_name.upper()}.{station_name}.s{start_date_str}.e{end_date_str}.{PRODUCT_VERSION}.parquet"
    )
    expected_path = os.path.join(path_campaign_name, "L0A", station_name, expected_name)
    assert res == expected_path


def test_get_l0b_fpath():
    """
    Test the naming and the path of the L0B file
    Note that this test needs "/data/test_dir_structure/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME/
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
        TEST_DATA_DIR,
        "test_dir_structure",
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
    res = io.get_l0b_fpath(ds, path_campaign_name, station_name)

    # Define expected results
    expected_name = f"L0B.{campaign_name.upper()}.{station_name}.s{start_date_str}.e{end_date_str}.{PRODUCT_VERSION}.nc"
    expected_path = os.path.join(path_campaign_name, "L0B", station_name, expected_name)
    assert res == expected_path


####--------------------------------------------------------------------------.


def test__check_glob_pattern():
    with pytest.raises(TypeError, match="Expect pattern as a string."):
        io._check_glob_pattern(1)

    with pytest.raises(ValueError, match="glob_pattern should not start with /"):
        io._check_glob_pattern("/1")


def test_get_raw_file_list():
    path_test_directory = os.path.join(TEST_DATA_DIR, "test_l0a_processing", "files")

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


def test__read_l0a():
    # create dummy dataframe
    data = [{"a": "1", "b": "2"}, {"a": "2", "b": "2", "c": "3"}]
    df = pd.DataFrame(data)

    # save dataframe to parquet file
    path_parquet_file = os.path.join(
        TEST_DATA_DIR,
        "test_dir_creation",
        "fake_data_sample.parquet",
    )
    df.to_parquet(path_parquet_file, compression="gzip")

    # read written parquet file
    df_written = io._read_l0a(path_parquet_file, False)

    assert df.equals(df_written)


def test_read_l0a_dataframe():
    list_of_parquet_file_paths = list()

    for i in [0, 1]:
        # create dummy dataframe
        data = [{"a": "1", "b": "2", "c": "3"}, {"a": "2", "b": "2", "c": "3"}]
        df = pd.DataFrame(data).set_index("a")
        df["time"] = pd.Timestamp.now()

        # save dataframe to parquet file
        path_parquet_file = os.path.join(
            TEST_DATA_DIR,
            "test_dir_creation",
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
    df_written = io.read_l0a_dataframe(list_of_parquet_file_paths, False)

    # Create lists
    df_concatenate_list = df_concatenate.values.tolist()
    df_written_list = df_written.values.tolist()

    # Compare lists
    comparison = df_written_list == df_concatenate_list

    assert comparison
