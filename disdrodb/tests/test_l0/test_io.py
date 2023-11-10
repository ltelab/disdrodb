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

import os

import pandas as pd
import pytest

from disdrodb.api.path import define_campaign_dir
from disdrodb.l0 import io
from disdrodb.tests.conftest import create_fake_raw_data_file

####--------------------------------------------------------------------------.


def test__check_glob_pattern():
    with pytest.raises(TypeError, match="Expect pattern as a string."):
        io._check_glob_pattern(1)

    with pytest.raises(ValueError, match="glob_pattern should not start with /"):
        io._check_glob_pattern("/1")


def test_get_raw_filepaths(tmp_path):
    # Define station info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    glob_pattern = "*.txt"
    raw_dir = define_campaign_dir(
        base_dir=base_dir, product="RAW", data_source=data_source, campaign_name=campaign_name
    )
    # Add fake data files
    for filename in ["file1.txt", "file2.txt"]:
        _ = create_fake_raw_data_file(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            filename=filename,
        )

    # Test that the function returns the correct number of files in debugging mode
    filepaths = io.get_raw_filepaths(
        raw_dir=raw_dir,
        station_name=station_name,
        glob_patterns=glob_pattern,
        debugging_mode=True,
    )
    assert len(filepaths) == 2  # max(2, 3)

    # Test that the function returns the correct number of files in normal mode
    filepaths = io.get_raw_filepaths(raw_dir=raw_dir, station_name=station_name, glob_patterns="*.txt")
    assert len(filepaths) == 2

    # Test that the function raises an error if the glob_patterns is not a str or list
    with pytest.raises(ValueError, match="'glob_patterns' must be a str or list of strings."):
        io.get_raw_filepaths(raw_dir=raw_dir, station_name=station_name, glob_patterns=1)

    # Test that the function raises an error if no files are found
    with pytest.raises(ValueError):
        io.get_raw_filepaths(
            raw_dir=raw_dir,
            station_name=station_name,
            glob_patterns="*.csv",
        )


####--------------------------------------------------------------------------.


def test__read_l0a(tmp_path):
    # create dummy dataframe
    data = [{"a": "1", "b": "2"}, {"a": "2", "b": "2", "c": "3"}]
    df = pd.DataFrame(data)

    # save dataframe to parquet file
    filepath = os.path.join(tmp_path, "fake_data_sample.parquet")
    df.to_parquet(filepath, compression="gzip")

    # read written parquet file
    df_written = io._read_l0a(filepath, False)

    assert df.equals(df_written)


def test_read_l0a_dataframe(tmp_path):
    filepaths = list()

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
    df_written = io.read_l0a_dataframe(filepaths, False)

    # Create lists
    df_concatenate_list = df_concatenate.values.tolist()
    df_written_list = df_written.values.tolist()

    # Compare lists
    comparison = df_written_list == df_concatenate_list

    assert comparison
