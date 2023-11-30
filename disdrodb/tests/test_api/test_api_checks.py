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
"""Test DISDRODB API checks utility."""


import os

import pytest

from disdrodb import __root_path__
from disdrodb.api.checks import (
    check_base_dir,
    check_path,
    check_path_is_a_directory,
    check_processed_dir,
    check_raw_dir,
    check_sensor_name,
    check_url,
)

TEST_DATA_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data")


def test_check_path():
    # Test a valid path
    path = os.path.abspath(__file__)
    assert check_path(path) is None

    # Test an invalid path
    path = "/path/that/does/not/exist"
    with pytest.raises(FileNotFoundError):
        check_path(path)


def test_check_url():
    # Test with valid URLs
    assert check_url("https://www.example.com") is True
    assert check_url("http://example.com/path/to/file.html?param=value") is True
    assert check_url("www.example.com") is True
    assert check_url("example.com") is True

    # Test with invalid URLs
    assert check_url("ftp://example.com") is False
    assert check_url("htp://example.com") is False
    assert check_url("http://example.com/path with spaces") is False


def test_check_base_dir():
    from pathlib import Path

    base_dir = os.path.join("path", "to", "DISDRODB")
    assert check_base_dir(base_dir) == base_dir

    assert check_base_dir(Path(base_dir)) == base_dir

    with pytest.raises(ValueError):
        check_base_dir("/path/to/DISDRO")


def test_check_sensor_name():
    sensor_name = "wrong_sensor_name"

    # Test with an unknown device
    with pytest.raises(ValueError):
        check_sensor_name(sensor_name)

    # Test with a woronf type
    with pytest.raises(TypeError):
        check_sensor_name(123)


def test_check_path_is_a_directory(tmp_path):
    base_dir = tmp_path / "DISDRODB"
    base_dir.mkdir(parents=True, exist_ok=True)
    check_path_is_a_directory(str(base_dir))


def test_check_raw_dir():
    # Set variables
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
    # Set paths
    raw_dir = os.path.join(
        TEST_DATA_DIR,
        "test_dir_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )

    assert check_raw_dir(raw_dir, station_name=station_name) == raw_dir


def test_check_processed_dir(tmp_path):
    # Check correct path
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "DATA_SOURCE" / "CAMPAIGN_NAME"
    processed_dir.mkdir(parents=True, exist_ok=True)

    assert check_processed_dir(str(processed_dir)) == str(processed_dir)

    # Check wrong type raises error
    with pytest.raises(TypeError):
        check_processed_dir(1)

    # Check wrong path (Raw)
    processed_dir = tmp_path / "DISDRODB" / "Raw" / "DATA_SOURCE" / "CAMPAIGN_NAME"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (only data_source)
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "DATA_SOURCE"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (only Processed)
    processed_dir = tmp_path / "DISDRODB" / "Processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (station_dir)
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "DATA_SOURCE" / "CAMPAIGN_NAME" / "data" / "station_name"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (lowercase data_source)
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "data_source" / "CAMPAIGN_NAME"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (lowercase data_source)
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "DATA_SOURCE" / "campaign_name"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))
