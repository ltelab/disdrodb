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
"""Test Metadata Info Extraction."""

import os

from disdrodb.metadata.info import get_archive_metadata_key_value
from disdrodb.utils.yaml import write_yaml


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


def test_get_archive_metadata_key_value(tmp_path):
    expected_result = []

    base_dir = os.path.join(tmp_path, "DISDRODB")
    # Test 1 : one config file
    yaml_file_name = "station_1.yml"
    expected_key = "key1"
    expected_value = "value1"
    data_source = "data_source"
    campaign_name = "campaign_name"

    yaml_dict = {expected_key: expected_value}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = get_archive_metadata_key_value(key=expected_key, base_dir=base_dir)
    expected_result.append((data_source, campaign_name, os.path.splitext(yaml_file_name)[0], expected_value))

    assert sorted(result) == sorted(expected_result)

    # Test 2 : two config files
    yaml_file_name = "station_2.yml"
    expected_key = "key1"
    expected_value = "value1"
    data_source = "data_source"
    campaign_name = "campaign_name"

    yaml_dict = {expected_key: expected_value}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = get_archive_metadata_key_value(key=expected_key, base_dir=base_dir)
    expected_result.append((data_source, campaign_name, os.path.splitext(yaml_file_name)[0], expected_value))

    assert sorted(result) == sorted(expected_result)

    # Test 3: test tuple
    yaml_file_name = "station_3.yml"
    expected_key = "key1"
    expected_value = "value1"
    data_source = "data_source"
    campaign_name = "campaign_name"
    yaml_dict = {expected_key: expected_value}
    create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    result = get_archive_metadata_key_value(key=expected_key, base_dir=base_dir, return_tuple=False)
    expected_result.append((data_source, campaign_name, os.path.splitext(yaml_file_name)[0], expected_value))
    expected_result = [item[3] for item in expected_result]

    assert sorted(result) == sorted(expected_result)
