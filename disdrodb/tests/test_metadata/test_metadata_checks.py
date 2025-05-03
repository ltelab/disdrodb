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
"""Check DISDRODB Metadata Archive files."""
import pytest

from disdrodb.api.configs import available_sensor_names
from disdrodb.l0.l0_reader import available_readers
from disdrodb.metadata.checks import (
    _check_metadata_values,
    check_archive_metadata_campaign_name,
    check_archive_metadata_compliance,
    check_archive_metadata_data_source,
    check_archive_metadata_geolocation,
    check_archive_metadata_keys,
    check_archive_metadata_reader,
    check_archive_metadata_sensor_name,
    check_archive_metadata_station_name,
    check_metadata_geolocation,
    identify_empty_metadata_keys,
    identify_missing_metadata_coords,
)
from disdrodb.metadata.standards import get_valid_metadata_keys
from disdrodb.tests.conftest import create_fake_metadata_file
from disdrodb.utils.yaml import read_yaml, write_yaml


def test_check_metadata_geolocation():
    # Test missing longitude and latitude
    metadata = {"platform_type": "fixed"}
    with pytest.raises(ValueError):
        check_metadata_geolocation(metadata)

    # Test non-numeric longitude
    metadata = {"longitude": "not_a_number", "latitude": 20, "platform_type": "fixed"}
    with pytest.raises(TypeError):
        check_metadata_geolocation(metadata)

    # Test non-numeric latitude
    metadata = {"longitude": 10, "latitude": "not_a_number", "platform_type": "fixed"}
    with pytest.raises(TypeError):
        check_metadata_geolocation(metadata)

    # Test mobile platform with wrong coordinates
    metadata = {"longitude": 10, "latitude": 20, "platform_type": "mobile"}
    with pytest.raises(ValueError):
        check_metadata_geolocation(metadata)

    # Test fixed platform with missing latitude
    metadata = {"longitude": 10, "latitude": -9999, "platform_type": "fixed"}
    with pytest.raises(ValueError):
        check_metadata_geolocation(metadata)

    # Test fixed platform with missing longitude
    metadata = {"longitude": -9999, "latitude": 20, "platform_type": "fixed"}
    with pytest.raises(ValueError):
        check_metadata_geolocation(metadata)

    # Test invalid longitude value
    metadata = {"longitude": 200, "latitude": 20, "platform_type": "fixed"}
    with pytest.raises(ValueError):
        check_metadata_geolocation(metadata)

    # Test invalid latitude value
    metadata = {"longitude": 10, "latitude": -100, "platform_type": "fixed"}
    with pytest.raises(ValueError):
        check_metadata_geolocation(metadata)

    # Test valid metadata
    metadata = {"longitude": 10, "latitude": 20, "platform_type": "fixed"}
    assert check_metadata_geolocation(metadata) is None


def test_identify_empty_metadata_keys(tmp_path, capsys):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"
    metadata_dict = {"key1": "value1"}
    metadata_filepath = create_fake_metadata_file(metadata_dir, metadata_dict=metadata_dict)

    # Test the key is empty -> print statement with the key name
    tested_key = "key2"
    identify_empty_metadata_keys([metadata_filepath], keys=tested_key)
    captured = capsys.readouterr()
    assert tested_key in str(captured.out)

    # Test the key is not empty -> no print statement
    tested_key = "key1"
    identify_empty_metadata_keys([metadata_filepath], [tested_key])
    captured = capsys.readouterr()
    assert not captured.out


def test_check_archive_metadata_keys(tmp_path):
    """Test check on correct archive."""
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    # Test 1: Correct metadata key
    valid_metadata_keys = get_valid_metadata_keys()
    metadata_dict = {i: "value1" for i in valid_metadata_keys}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)

    is_valid = check_archive_metadata_keys(metadata_dir)
    assert is_valid

    # Test 2 : Wrong metadata key
    metadata_dict = {"should_not_be_found": "value"}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_keys(metadata_dir)
    assert not is_valid

    # Test 3 : Check missing metadata key
    metadata_dict = {}
    metadata_filepath = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)
    metadata_dict = read_yaml(metadata_filepath)
    metadata_dict.pop("data_source")
    write_yaml(metadata_dict, metadata_filepath)
    is_valid = check_archive_metadata_keys(metadata_dir)
    assert not is_valid


def test_check_archive_metadata_valid_values(tmp_path):
    """Test that None is an invalid value."""
    metadata_dict = {"key_is_None": None}
    with pytest.raises(ValueError):
        _check_metadata_values(metadata_dict)


def test_check_archive_metadata_campaign_name(tmp_path):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    # Test 1 : Correct campaign_name metadata key
    campaign_name = "CAMPAIGN_NAME"
    metadata_dict = {"campaign_name": campaign_name}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, campaign_name=campaign_name, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_campaign_name(metadata_dir)
    assert is_valid

    # Test 2 : Empty campaign_name
    campaign_name = "CAMPAIGN_NAME"
    metadata_dict = {"campaign_name": ""}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, campaign_name=campaign_name, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_campaign_name(metadata_dir)
    assert not is_valid

    # Test 3 : Wrong campaign_name
    campaign_name = "CAMPAIGN_NAME"
    metadata_dict = {"campaign_name": "ANOTHER_CAMPAIGN_NAME"}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, campaign_name=campaign_name, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_campaign_name(metadata_dir)
    assert not is_valid

    # Test 4 : Missing campaign_name
    campaign_name = "CAMPAIGN_NAME"
    metadata_filepath = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        campaign_name=campaign_name,
    )
    metadata_dict = read_yaml(metadata_filepath)
    metadata_dict.pop("campaign_name", None)
    write_yaml(metadata_dict, metadata_filepath)
    is_valid = check_archive_metadata_campaign_name(metadata_dir)
    assert not is_valid


def test_check_archive_metadata_data_source(tmp_path):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    # Test 1 : Correct data_source metadata key
    data_source = "DATA_SOURCE"
    metadata_dict = {"data_source": data_source}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, data_source=data_source, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_data_source(metadata_dir)
    assert is_valid

    # Test 2 : Empty data_source metadata key
    data_source = "DATA_SOURCE"
    metadata_dict = {"data_source": ""}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, data_source=data_source, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_data_source(metadata_dir)
    assert not is_valid

    # Test 3 : Wrong data_source
    data_source = "DATA_SOURCE"
    metadata_dict = {"data_source": "ANOTHER_DATA_SOURCE"}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, data_source=data_source, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_data_source(metadata_dir)
    assert not is_valid

    # Test 4 : Missing data_source
    data_source = "DATA_SOURCE"
    metadata_filepath = create_fake_metadata_file(metadata_dir=metadata_dir, data_source=data_source)
    metadata_dict = read_yaml(metadata_filepath)
    metadata_dict.pop("data_source", None)
    write_yaml(metadata_dict, metadata_filepath)
    is_valid = check_archive_metadata_data_source(metadata_dir)
    assert not is_valid


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_archive_metadata_sensor_name(tmp_path, sensor_name):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    # Test 1 : Correct sensor_name metadata key
    metadata_dict = {"sensor_name": sensor_name}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_sensor_name(metadata_dir)
    assert is_valid

    # Test 2 : Wrong sensor_name metadata key
    metadata_dict = {"sensor_name": ""}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_sensor_name(metadata_dir)
    assert not is_valid


def test_check_archive_metadata_station_name(tmp_path):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    # Test 1 : Correct station_name metadata key
    station_name = "station_name"
    metadata_dict = {"station_name": station_name}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict, station_name=station_name)
    is_valid = check_archive_metadata_station_name(metadata_dir)
    assert is_valid

    # Test 2 : Empty station_name metadata key
    metadata_dict = {"station_name": ""}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict, station_name=station_name)
    is_valid = check_archive_metadata_station_name(metadata_dir)
    assert not is_valid

    # Test 3 : Wrong station_name
    station_name = "STATION_NAME"
    metadata_dict = {"station_name": "ANOTHER_STATION_NAME"}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, station_name=station_name, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_station_name(metadata_dir)
    assert not is_valid

    # Test 4 : Missing station_name
    station_name = "STATION_NAME"
    metadata_filepath = create_fake_metadata_file(metadata_dir=metadata_dir, station_name=station_name)
    metadata_dict = read_yaml(metadata_filepath)
    metadata_dict.pop("station_name", None)
    write_yaml(metadata_dict, metadata_filepath)
    is_valid = check_archive_metadata_station_name(metadata_dir)
    assert not is_valid

    # Test 5 : Invalid station_name value type
    metadata_dict = {"station_name": 2}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict, station_name=station_name)
    is_valid = check_archive_metadata_station_name(metadata_dir)
    assert not is_valid


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_archive_metadata_reader(tmp_path, sensor_name):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    list_readers = available_readers(sensor_name)

    # Test 1 : Correct reader metadata key
    reader_reference = list_readers[0]
    metadata_dict = {"reader": f"{reader_reference}", "sensor_name": sensor_name}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_reader(metadata_dir)
    assert is_valid

    # Test 2 : Wrong reader metadata key
    metadata_dict = {"reader": "", "sensor_name": sensor_name}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_reader(metadata_dir)
    assert not is_valid

    # Test 3 : Wrong reader metadata key
    metadata_dict = {"reader": "dummy/dummy", "sensor_name": sensor_name}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)
    is_valid = check_archive_metadata_reader(metadata_dir)
    assert not is_valid


def test_check_archive_metadata_compliance(tmp_path):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    # We check only the failure, the success are tested in the above tests.
    metadata_dict = {"reader": ""}
    _ = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)

    # Test does not raise error !
    result = check_archive_metadata_compliance(metadata_dir, raise_error=False)
    assert result is False

    # Test it raise error
    with pytest.raises(ValueError):
        result = check_archive_metadata_compliance(metadata_dir, raise_error=True)


@pytest.mark.parametrize("platform_type", ["mobile", "fixed"])
@pytest.mark.parametrize("latlon_value", [0, 500, -9999, -99991, "bad_type"])
def test_check_archive_metadata_geolocation(tmp_path, latlon_value, platform_type):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    metadata_dict = {"longitude": latlon_value, "latitude": latlon_value, "platform_type": platform_type}
    _ = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict=metadata_dict,
    )
    is_valid = check_archive_metadata_geolocation(metadata_dir)
    if (platform_type == "mobile" and latlon_value == -9999) or (platform_type != "mobile" and latlon_value == 0):
        assert is_valid
    else:
        assert not is_valid


def test_identify_missing_metadata_coords(tmp_path):
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    # Test correct coordinates
    metadata_dict = {"longitude": 170, "latitude": 80, "platform_type": "fixed"}
    metadata_filepath = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict=metadata_dict,
    )

    function_return = identify_missing_metadata_coords([metadata_filepath])
    assert function_return is None

    # Test bad coordinates
    metadata_dict = {"longitude": "8r0", "latitude": "170", "platform_type": "fixed"}
    metadata_filepath = create_fake_metadata_file(metadata_dir=metadata_dir, metadata_dict=metadata_dict)
    with pytest.raises(TypeError):
        identify_missing_metadata_coords([metadata_filepath])
