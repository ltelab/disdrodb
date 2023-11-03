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

import os

import pytest
import yaml

from disdrodb import __root_path__
from disdrodb.api.configs import available_sensor_names
from disdrodb.l0.l0_reader import available_readers
from disdrodb.metadata.check_metadata import (
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
from disdrodb.utils.yaml import read_yaml, write_yaml

TEST_DATA_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data")


def test_check_metadata_geolocation():
    # Test missing longitude and latitude
    with pytest.raises(ValueError):
        metadata = {"platform_type": "fixed"}
        check_metadata_geolocation(metadata)

    # Test non-numeric longitude
    with pytest.raises(TypeError):
        metadata = {"longitude": "not_a_number", "latitude": 20, "platform_type": "fixed"}
        check_metadata_geolocation(metadata)

    # Test non-numeric latitude
    with pytest.raises(TypeError):
        metadata = {"longitude": 10, "latitude": "not_a_number", "platform_type": "fixed"}
        check_metadata_geolocation(metadata)

    # Test mobile platform with wrong coordinates
    with pytest.raises(ValueError):
        metadata = {"longitude": 10, "latitude": 20, "platform_type": "mobile"}
        check_metadata_geolocation(metadata)

    # Test fixed platform with missing latitude
    with pytest.raises(ValueError):
        metadata = {"longitude": 10, "latitude": -9999, "platform_type": "fixed"}
        check_metadata_geolocation(metadata)

    # Test fixed platform with missing longitude
    with pytest.raises(ValueError):
        metadata = {"longitude": -9999, "latitude": 20, "platform_type": "fixed"}
        check_metadata_geolocation(metadata)

    # Test invalid longitude value
    with pytest.raises(ValueError):
        metadata = {"longitude": 200, "latitude": 20, "platform_type": "fixed"}
        check_metadata_geolocation(metadata)

    # Test invalid latitude value
    with pytest.raises(ValueError):
        metadata = {"longitude": 10, "latitude": -100, "platform_type": "fixed"}
        check_metadata_geolocation(metadata)

    # Test valid metadata
    metadata = {"longitude": 10, "latitude": 20, "platform_type": "fixed"}
    assert check_metadata_geolocation(metadata) is None


def create_fake_metadata_file(
    tmp_path, yaml_file_name, yaml_dict, data_source="data_source", campaign_name="campaign_name"
):
    subfolder_path = tmp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)
    file_path = os.path.join(subfolder_path, yaml_file_name)
    # create a fake yaml file in temp folder
    write_yaml(yaml_dict, file_path)
    assert os.path.exists(file_path)
    return file_path


def test_identify_missing_metadata_keys(tmp_path, capsys):
    yaml_file_name = "test.yml"
    yaml_dict = {"key1": "value1"}

    fake_metadata_file_path = create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict)

    # Test the key is empty -> print statement with the key name
    tested_key = "key2"
    identify_empty_metadata_keys([fake_metadata_file_path], [tested_key])
    captured = capsys.readouterr()
    assert tested_key in str(captured.out)

    # Test the key is not empty -> no print statement
    tested_key = "key1"
    identify_empty_metadata_keys([fake_metadata_file_path], [tested_key])
    captured = capsys.readouterr()
    assert not captured.out


def test_check_archive_metadata_keys(tmp_path):
    base_dir = os.path.join(tmp_path, "DISDRODB")
    # Test 1 : create a correct metadata file
    # Get the list of valid metadata keys
    list_of_valid_metadata_keys = get_valid_metadata_keys()
    yaml_file_name = "station_1.yml"
    yaml_dict = {i: "value1" for i in list_of_valid_metadata_keys}
    data_source = "data_source"
    campaign_name = "campaign_name"
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_keys(base_dir)
    assert result is True

    # Test 2 : add a wrong metadata key file
    yaml_file_name = "station_2.yml"
    expected_key = "should_not_be_found"
    expected_value = "value1"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {expected_key: expected_value}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_keys(base_dir)
    # assert result is False


def test_check_archive_metadata_campaign_name(tmp_path):
    base_dir = os.path.join(tmp_path, "DISDRODB")

    # Test 1 : create a correct metadata file
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"campaign_name": campaign_name}
    create_fake_metadata_file(
        tmp_path=tmp_path,
        yaml_file_name=yaml_file_name,
        yaml_dict=yaml_dict,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    result = check_archive_metadata_campaign_name(base_dir)
    assert result is True

    # Test 2 : create a wrong metadata file
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"campaign_name": ""}
    create_fake_metadata_file(
        tmp_path=tmp_path,
        yaml_file_name=yaml_file_name,
        yaml_dict=yaml_dict,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    result = check_archive_metadata_campaign_name(base_dir)
    assert result is False


def test_check_archive_metadata_data_source(tmp_path):
    base_dir = os.path.join(tmp_path, "DISDRODB")

    # Test 1 : create a correct metadata file
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"data_source": data_source}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_data_source(base_dir)
    assert result is True

    # Test 2 : create a wrong metadata file
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"data_source": ""}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_data_source(base_dir)
    assert result is False


def test_check_archive_metadata_sensor_name(tmp_path):
    sensor_names = available_sensor_names(product_level="l0")
    base_dir = os.path.join(tmp_path, "DISDRODB")

    # Test 1 : create a correct metadata file
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    sensor_name = sensor_names[0]
    yaml_dict = {"sensor_name": sensor_name}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_sensor_name(base_dir)
    assert result is True

    # Test 2 : create a wrong metadata file
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"sensor_name": ""}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_sensor_name(base_dir)
    assert result is False


def test_check_archive_metadata_station_name(tmp_path):
    base_dir = os.path.join(tmp_path, "DISDRODB")

    # Test 1 : create a correct metadata file
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"station_name": os.path.splitext(yaml_file_name)[0]}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_station_name(base_dir)
    assert result is True

    # Test 2 : create a wrong metadata file
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"station_name": ""}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_station_name(base_dir)
    assert result is False


def test_check_archive_metadata_reader(tmp_path):
    base_dir = os.path.join(tmp_path, "DISDRODB")

    # Test 1 : create a correct metadata file
    list_readers = available_readers()
    yaml_file_name = "station_1.yml"
    campaign_name = "campaign_name"
    data_source = list(list_readers.keys())[0]
    reader_names = list_readers[data_source]
    reader_name = reader_names[0]
    yaml_dict = {"reader": f"{data_source}/{reader_name}"}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_reader(base_dir)
    assert result is True

    # Test 2 : create a wrong metadata file
    yaml_file_name = "station_1.yml"
    campaign_name = "campaign_name"
    yaml_dict = {"reader": ""}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_reader(base_dir)
    assert result is False


def test_check_archive_metadata_compliance(tmp_path):
    base_dir = os.path.join(tmp_path, "DISDRODB")

    # We check only the failure, the success are tested in the above tests.
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"reader": ""}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_compliance(base_dir)
    assert result is False


def test_check_archive_metadata_geolocation(tmp_path):
    # We check only the failure, the success are tested in the above test.
    base_dir = os.path.join(tmp_path, "DISDRODB")
    yaml_file_name = "station_1.yml"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {"reader": ""}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = check_archive_metadata_geolocation(base_dir)
    assert result is False


def test_read_yaml():
    # test based on files under tests\data\test_check_metadata

    # Test reading a valid YAML file
    valid_yaml_attrs = {"key1": "value1", "key2": "value2"}
    yaml_temp_path = os.path.join(TEST_DATA_DIR, "test_check_metadata", "valid.yaml")
    assert read_yaml(yaml_temp_path) == valid_yaml_attrs

    # Test reading a non-existent YAML file
    yaml_temp_path_non_existent = os.path.join(TEST_DATA_DIR, "test_check_metadata", "non_existent.yaml")
    with pytest.raises(FileNotFoundError):
        read_yaml(yaml_temp_path_non_existent)

    # Test reading a YAML file with invalid syntax
    yaml_temp_path_nvalid = os.path.join(TEST_DATA_DIR, "test_check_metadata", "invalid.yaml")
    with pytest.raises(yaml.YAMLError):
        read_yaml(yaml_temp_path_nvalid)


def test_identify_missing_metadata_coords():
    yaml_temp_path_valid = [os.path.join(TEST_DATA_DIR, "test_check_metadata", "valid_coords.yaml")]

    function_return = identify_missing_metadata_coords(yaml_temp_path_valid)
    assert function_return is None

    yaml_temp_path_invalid = [os.path.join(TEST_DATA_DIR, "test_check_metadata", "invalid_coords.yaml")]

    with pytest.raises(TypeError):
        identify_missing_metadata_coords(yaml_temp_path_invalid)
