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
"""Test DISDRODB L0 readers routines."""

import pytest

import disdrodb
from disdrodb.l0.l0_reader import (
    available_readers,
    check_metadata_reader,
    check_reader_arguments,
    check_reader_exists,
    check_reader_reference,
    check_software_readers,
    define_reader_path,
    get_reader,
    get_station_reader,
)
from disdrodb.tests.conftest import create_fake_metadata_file
from disdrodb.utils.yaml import read_yaml, write_yaml

# Some test are based on the following reader:
DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "EPFL_2009"
READER_REFERENCE = f"{DATA_SOURCE}/{CAMPAIGN_NAME}"
SENSOR_NAME = "PARSIVEL"


def test_check_reader_reference():
    """Test check_reader_reference function."""
    assert check_reader_reference(READER_REFERENCE) == READER_REFERENCE


def test_check_reader_exists():
    """Test check_reader_exists function."""
    # Check existing reader do not raise error
    check_reader_exists(READER_REFERENCE, SENSOR_NAME)

    # Check unexisting reader
    with pytest.raises(ValueError):
        check_reader_exists("EPFL", SENSOR_NAME)


class TestCheckReaderArguments:
    def test_valid_reader(self):
        """Should not raise error for valid reader signature."""

        def reader(filepath, logger=None):
            return "dummy"

        # No exception expected
        check_reader_arguments(reader)

    def test_missing_parameters(self):
        """Should raise ValueError when arguments are incorrect."""

        def reader(bad_arg):
            return "dummy"

        with pytest.raises(ValueError) as excinfo:
            check_reader_arguments(reader)
        assert "following arguments" in str(excinfo.value)

    def test_logger_no_default(self):
        """Should raise ValueError when logger default is not specified."""

        def reader(filepath, logger):
            return "dummy"

        with pytest.raises(ValueError) as excinfo:
            check_reader_arguments(reader)
        assert "must have a default value" in str(excinfo.value)

    def test_logger_default_not_none(self):
        """Should raise ValueError when logger default is not None."""

        def reader(filepath, logger="not_none"):
            return "dummy"

        with pytest.raises(ValueError) as excinfo:
            check_reader_arguments(reader)
        assert "must be None" in str(excinfo.value)


class TestAvailableReaders:
    def test_list_of_references(self):
        """Should return a list of reader references as strings."""
        readers_references = available_readers(sensor_name=SENSOR_NAME)
        assert isinstance(readers_references, list)
        assert all(isinstance(reader_reference, str) for reader_reference in readers_references)
        assert "EPFL/EPFL_2009" in readers_references
        assert "NASA/MC3E" in readers_references

    def test_filter_references_by_source(self):
        """Should filter reference list by given data source."""
        readers_references = available_readers(sensor_name=SENSOR_NAME, data_sources="EPFL")
        assert isinstance(readers_references, list)
        assert all(isinstance(reader_reference, str) for reader_reference in readers_references)
        assert "EPFL/EPFL_2009" in readers_references
        assert "NASA/MC3E" not in readers_references  # Filtered out

    def test_list_of_paths(self):
        """Should return a list of reader file paths as strings."""
        readers_paths = available_readers(sensor_name=SENSOR_NAME, return_path=True)
        assert isinstance(readers_paths, list)
        assert all(isinstance(reader_path, str) for reader_path in readers_paths)
        assert define_reader_path(reader_reference="EPFL/EPFL_2009", sensor_name=SENSOR_NAME) in readers_paths
        assert define_reader_path(reader_reference="NASA/MC3E", sensor_name=SENSOR_NAME) in readers_paths

    def test_filter_paths_by_source(self):
        """Should filter path list by given data source."""
        readers_paths = available_readers(sensor_name=SENSOR_NAME, data_sources="EPFL", return_path=True)
        assert isinstance(readers_paths, list)
        assert all(isinstance(reader_path, str) for reader_path in readers_paths)
        assert define_reader_path(reader_reference="EPFL/EPFL_2009", sensor_name=SENSOR_NAME) in readers_paths
        assert define_reader_path(reader_reference="NASA/MC3E", sensor_name=SENSOR_NAME) not in readers_paths


class TestCheckMetadataReader:
    def test_missing_reader_key_empty_metadata(self):
        """Should raise ValueError when 'reader' key missing in metadata dictionary."""
        with pytest.raises(ValueError, match="The `reader` key is not specified in the metadata"):
            check_metadata_reader({})

    def test_missing_reader_key(self):
        """Should raise ValueError when 'reader' key missing even if other keys present."""
        with pytest.raises(ValueError, match="The `reader` key is not specified in the metadata"):
            check_metadata_reader({"another_key": "whatever"})

    def test_invalid_sensor_name(self):
        """Should raise ValueError for an invalid sensor_name."""
        with pytest.raises(ValueError, match="'Invalid_sensor' is not a valid sensor_name."):
            check_metadata_reader({"reader": READER_REFERENCE, "sensor_name": "Invalid_sensor"})

    def test_missing_sensor_name(self):
        """Should raise ValueError when sensor_name is not specified."""
        with pytest.raises(ValueError, match="The `sensor_name` is not specified in the metadata."):
            check_metadata_reader({"reader": None})

    def test_invalid_reader_reference_does_not_exist(self):
        """Should raise ValueError when reader_reference does not exist."""
        with pytest.raises(ValueError, match="PARSIVEL reader 'invalid_reader' does not exists."):
            check_metadata_reader({"reader": "invalid_reader", "sensor_name": SENSOR_NAME})

    def test_reader_key_is_none(self):
        """Should raise TypeError when reader_reference is None and not a string."""
        with pytest.raises(TypeError, match="`reader_reference` is None. Specify the reader reference name"):
            check_metadata_reader({"reader": None, "sensor_name": SENSOR_NAME})

    def test_reader_key_not_string(self):
        """Should raise TypeError when reader_reference is not a string type."""
        with pytest.raises(TypeError, match="`reader_reference` must be a string. Got type"):
            check_metadata_reader({"reader": 2, "sensor_name": SENSOR_NAME})

    def test_reader_key_empty_string(self):
        """Should raise ValueError when reader_reference is an empty string."""
        with pytest.raises(ValueError, match="`reader_reference` is an empty string"):
            check_metadata_reader({"reader": "", "sensor_name": SENSOR_NAME})

    def test_reader_reference_too_many_components(self):
        """Should raise ValueError when reader_reference has more than two components."""
        with pytest.raises(ValueError, match="`reader_reference` expects to be composed by maximum"):
            check_metadata_reader({"reader": "ONE/TWO/THREE", "sensor_name": SENSOR_NAME})

    def test_valid_metadata(self):
        """Should not raise error for valid metadata."""
        # No exception expected
        check_metadata_reader({"reader": READER_REFERENCE, "sensor_name": SENSOR_NAME})


# import pathlib
# tmp_path = pathlib.Path("/tmp/10")
def test_get_station_reader(tmp_path):
    """Test retrieve reader from metadata file."""
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    station_name = "station_name"

    metadata_dict = {"reader": READER_REFERENCE, "sensor_name": SENSOR_NAME}

    metadata_filepath = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=station_name,
    )
    result = get_station_reader(
        metadata_archive_dir=metadata_archive_dir,
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=station_name,
    )
    assert callable(result)

    # Check function available from package root
    result = disdrodb.get_station_reader(
        metadata_archive_dir=metadata_archive_dir,
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=station_name,
    )
    assert callable(result)

    # Assert raise error if not reader key in metadata
    metadata_dict = read_yaml(metadata_filepath)
    metadata_dict.pop("reader", None)
    write_yaml(metadata_dict, metadata_filepath)

    with pytest.raises(ValueError, match="The `reader` key is not specified in the metadata"):
        get_station_reader(
            metadata_archive_dir=metadata_archive_dir,
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=station_name,
        )


def test_get_reader(tmp_path):
    """Test get_reader function."""
    reader = get_reader(reader_reference=READER_REFERENCE, sensor_name=SENSOR_NAME)
    assert callable(reader)

    # Check function available from package root
    reader = disdrodb.get_reader(reader_reference=READER_REFERENCE, sensor_name=SENSOR_NAME)
    assert callable(reader)


def test_check_software_readers():
    """Check validity of all software readers."""
    check_software_readers()
