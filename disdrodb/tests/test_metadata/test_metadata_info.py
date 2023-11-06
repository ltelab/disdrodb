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


from disdrodb.metadata.info import get_archive_metadata_key_value
from disdrodb.tests.conftest import create_fake_metadata_file


def test_get_archive_metadata_key_value(tmp_path):
    base_dir = tmp_path / "DISDRODB"

    expected_result = []

    # Test 1 : one config file
    expected_key = "key1"
    expected_value = "value1"
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name1"

    metadata_dict = {expected_key: expected_value}
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    result = get_archive_metadata_key_value(key=expected_key, return_tuple=True, base_dir=base_dir)
    expected_result.append((data_source, campaign_name, station_name, expected_value))

    assert sorted(result) == sorted(expected_result)

    # Test 2 : two config files
    expected_key = "key1"
    expected_value = "value1"
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name2"
    metadata_dict = {expected_key: expected_value}
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    result = get_archive_metadata_key_value(key=expected_key, base_dir=base_dir, return_tuple=True)
    expected_result.append((data_source, campaign_name, station_name, expected_value))

    assert sorted(result) == sorted(expected_result)
    assert len(result) == 2

    # Test 3: test tuple
    expected_key = "key1"
    expected_value = "value1"
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name3"

    metadata_dict = {expected_key: expected_value}
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    result = get_archive_metadata_key_value(key=expected_key, base_dir=base_dir, return_tuple=True)
    values = get_archive_metadata_key_value(key=expected_key, base_dir=base_dir, return_tuple=False)
    expected_result.append((data_source, campaign_name, station_name, expected_value))
    expected_values = [item[3] for item in result]

    assert sorted(values) == sorted(expected_values)
