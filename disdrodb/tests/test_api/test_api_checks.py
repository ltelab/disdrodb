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
    check_data_archive_dir,
    check_path,
    check_path_is_a_directory,
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
    assert check_url("https://www.example.com")
    assert check_url("http://example.com/path/to/file.html?param=value")
    assert check_url("www.example.com")
    assert check_url("example.com")

    # Test with invalid URLs
    assert not check_url("ftp://example.com")
    assert not check_url("htp://example.com")
    assert not check_url("http://example.com/path with spaces")


def test_check_data_archive_dir():
    from pathlib import Path

    data_archive_dir = os.path.join("path", "to", "DISDRODB")
    assert check_data_archive_dir(data_archive_dir) == data_archive_dir

    assert check_data_archive_dir(Path(data_archive_dir)) == data_archive_dir

    with pytest.raises(ValueError):
        check_data_archive_dir("/path/to/DISDRO")


def test_check_sensor_name():
    sensor_name = "wrong_sensor_name"

    # Test with an unknown device
    with pytest.raises(ValueError):
        check_sensor_name(sensor_name)

    # Test with a woronf type
    with pytest.raises(TypeError):
        check_sensor_name(123)


def test_check_path_is_a_directory(tmp_path):
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    data_archive_dir.mkdir(parents=True, exist_ok=True)
    check_path_is_a_directory(str(data_archive_dir))
    check_path_is_a_directory(data_archive_dir)
