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

from disdrodb.api.checks import (
    check_base_dir,
    check_path,
    check_sensor_name,
    check_url,
)


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
