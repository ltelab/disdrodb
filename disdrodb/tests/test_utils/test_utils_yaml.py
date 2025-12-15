# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
import yaml

from disdrodb import package_dir
from disdrodb.utils.yaml import read_yaml, write_yaml

TEST_DATA_DIR = os.path.join(package_dir, "tests", "data")


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


class TestWriteYAML:
    def test_write_yaml_creates_file_and_roundtrips(self, tmp_path):
        """It writes a dict to YAML and loads back correctly."""
        data = {"b": 2, "a": 1}
        filepath = tmp_path / "test.yaml"

        write_yaml(data, filepath)

        assert filepath.exists()
        with open(filepath) as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_write_yaml_respects_sort_keys(self, tmp_path):
        """It respects sort_keys when writing YAML."""
        data = {"b": 2, "a": 1}
        filepath_unsorted = tmp_path / "unsorted.yaml"
        filepath_sorted = tmp_path / "sorted.yaml"

        write_yaml(data, filepath_unsorted, sort_keys=False)
        unsorted_text = filepath_unsorted.read_text()

        write_yaml(data, filepath_sorted, sort_keys=True)
        sorted_text = filepath_sorted.read_text()

        # Roundtrip correctness
        assert yaml.safe_load(unsorted_text) == data
        assert yaml.safe_load(sorted_text) == data

        # Check ordering in YAML string
        assert sorted_text.find("a:") < sorted_text.find("b:")
