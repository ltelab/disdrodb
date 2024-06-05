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
"""Test YAML utility."""

import os

import pytest

from disdrodb import __root_path__
from disdrodb.utils.yaml import read_yaml

TEST_DATA_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data")


def test_read_yaml():
    # Test reading a YAML file get the expect types
    # - string, list, int, float, list[str], list[int], None
    dictionary = {
        "key1": "value1",
        "key2": 2,
        "key3": 3.0,
        "key4": ["value4"],
        "key5": [5],
        "key6": None,
        "key7": "",
    }
    filepath = os.path.join(TEST_DATA_DIR, "test_yaml", "valid.yaml")
    assert read_yaml(filepath) == dictionary

    # Test reading a non-existent YAML file
    non_existent_filepath = os.path.join(TEST_DATA_DIR, "non_existent.yaml")
    with pytest.raises(FileNotFoundError):
        read_yaml(non_existent_filepath)
