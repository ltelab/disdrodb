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
"""Test DISDRODB raw data compression."""


import os

import pytest

from disdrodb.utils.compression import _unzip_file, _zip_dir, compress_station_files


def create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name):
    """Create a station data directory with files inside it.

    station_name
    |-- dir1
        |-- file1.txt
        |-- dir1
            |-- file2.txt
    """

    data_dir = disdrodb_dir / "Raw" / data_source / campaign_name / "data" / station_name
    dir1 = data_dir / "dir1"
    dir2 = dir1 / "dir2"
    if not dir2.exists():
        dir2.mkdir(parents=True)

    file1_txt = dir1 / "file1.txt"
    file1_txt.touch()
    file2_txt = dir2 / "file2.txt"
    file2_txt.touch()


def test_files_compression(tmp_path):
    """Test compression of files in a directory."""

    disdrodb_dir = tmp_path / "DISDRODB"
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"

    # Directory that does not exist yet
    compress_station_files(disdrodb_dir, data_source, campaign_name, "station1", "zip")

    methods = ["zip", "gzip", "bzip2"]
    for i, method in enumerate(methods):
        station_name = f"test_station_name_{i}"
        create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name)
        compress_station_files(disdrodb_dir, data_source, campaign_name, station_name, method=method)

    # Directory with already compressed files
    station_name = "test_station_name_0"
    compress_station_files(disdrodb_dir, data_source, campaign_name, station_name, "zip")

    station_name = "test_station_name"
    create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name)
    with pytest.raises(ValueError):
        compress_station_files(disdrodb_dir, data_source, campaign_name, station_name, "unknown_compression_method")


def test_zip_unzip_directory(tmp_path):
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    file_path = dir_path / "test_file.txt"
    file_path.touch()

    zip_path = _zip_dir(dir_path)
    assert os.path.isfile(zip_path)

    unzip_path = tmp_path / "test_dir_unzipped"
    _unzip_file(zip_path, unzip_path)
    assert os.path.isdir(unzip_path)
