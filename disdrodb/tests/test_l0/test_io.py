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

TEST_DATA_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data")

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
