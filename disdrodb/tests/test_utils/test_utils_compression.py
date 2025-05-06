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
import pathlib

import pytest

from disdrodb.tests.conftest import create_fake_data_dir, create_fake_raw_data_file
from disdrodb.utils.compression import _zip_dir, compress_station_files, unzip_file


@pytest.mark.parametrize("method", ["zip", "gzip", "bzip2"])
def test_files_compression(tmp_path, method):
    """Test compression of files in a directory."""
    base_dir = tmp_path / "data" / "DISDRODB"
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "station_name"

    # Check raise an error if the directory does not yet exist
    with pytest.raises(ValueError):
        compress_station_files(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            method=method,
        )

    # Create fake data
    data_dir = create_fake_data_dir(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    data_dir = pathlib.Path(data_dir)
    dir1 = data_dir / "2020"
    dir2 = dir1 / "Jan"
    if not dir2.exists():
        dir2.mkdir(parents=True)

    file1_txt = dir1 / "file1.txt"
    file1_txt.touch()
    file2_txt = dir2 / "file2.txt"
    file2_txt.touch()

    # Compress files
    compress_station_files(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        method=method,
    )

    # Try to compress directory with already compressed files (skip=True)
    compress_station_files(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        method=method,
        skip=True,
    )

    # Try to compress directory with already compressed files (skip=False)
    with pytest.raises(ValueError):
        compress_station_files(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            method=method,
            skip=False,
        )

    # Try to compress with invalid method
    with pytest.raises(ValueError):
        compress_station_files(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            method="unknown_compression_method",
        )

    # Try to compress a netCDF file
    create_fake_raw_data_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        filename="test_data.nc",
    )
    with pytest.raises(ValueError):
        compress_station_files(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            method=method,
        )


def test_zip_unzip_directory(tmp_path):
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    filepath = dir_path / "test_file.txt"
    filepath.touch()

    zip_path = _zip_dir(dir_path)
    assert os.path.isfile(zip_path)

    unzip_path = tmp_path / "test_dir_unzipped"
    unzip_file(zip_path, unzip_path)
    assert os.path.isdir(unzip_path)
