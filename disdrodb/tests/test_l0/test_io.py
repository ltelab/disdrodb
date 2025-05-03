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


import pytest

from disdrodb.api.io import find_files
from disdrodb.api.path import define_campaign_dir
from disdrodb.l0.io import (
    _check_glob_pattern,
    get_raw_filepaths,
)
from disdrodb.tests.conftest import create_fake_raw_data_file

####--------------------------------------------------------------------------.


def test__check_glob_pattern():
    with pytest.raises(TypeError, match="Expect pattern as a string."):
        _check_glob_pattern(1)

    with pytest.raises(ValueError, match="glob_pattern should not start with /"):
        _check_glob_pattern("/1")


def test_get_raw_filepaths(tmp_path):
    # Define station info
    base_dir = tmp_path / "data" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    glob_pattern = "*.txt"
    raw_dir = define_campaign_dir(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
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
    filepaths = get_raw_filepaths(
        raw_dir=raw_dir,
        station_name=station_name,
        glob_patterns=glob_pattern,
        debugging_mode=True,
    )
    assert len(filepaths) == 2  # max(2, 3)

    # Test that the function returns the correct number of files in normal mode
    filepaths = get_raw_filepaths(raw_dir=raw_dir, station_name=station_name, glob_patterns="*.txt")
    assert len(filepaths) == 2

    # Test that the function raises an error if the glob_patterns is not a str or list
    with pytest.raises(ValueError, match="'glob_patterns' must be a str or list of strings."):
        get_raw_filepaths(raw_dir=raw_dir, station_name=station_name, glob_patterns=1)

    # Test that the function raises an error if no files are found
    with pytest.raises(ValueError):
        get_raw_filepaths(
            raw_dir=raw_dir,
            station_name=station_name,
            glob_patterns="*.csv",
        )


def test_get_l0a_filepaths(tmp_path):
    # Define station info
    base_dir = tmp_path / "data" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    # Test that the function raises an error if no files presenet
    with pytest.raises(ValueError):
        _ = find_files(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="L0A",
        )

    # Add fake data files
    for filename in ["file1.parquet", "file2.parquet"]:
        _ = create_fake_raw_data_file(
            base_dir=base_dir,
            product="L0A",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            filename=filename,
        )

    # Test that the function returns the correct number of files in debugging mode
    filepaths = find_files(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="L0A",
        debugging_mode=True,
    )
    assert len(filepaths) == 2  # max(2, 3)

    # Test that the function returns the correct number of files in normal mode
    filepaths = find_files(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="L0A",
    )
    assert len(filepaths) == 2


####--------------------------------------------------------------------------.
