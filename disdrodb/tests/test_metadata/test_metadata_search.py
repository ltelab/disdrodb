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
"""Test DISDRODB API metadata utility."""

import pytest

from disdrodb.metadata.search import (
    _get_list_all_metadata,
    _get_list_metadata_with_data,
    get_list_metadata,
)
from disdrodb.tests.conftest import (
    create_fake_metadata_file,
    create_fake_raw_data_file,
)


def test__get_list_all_metadata(tmp_path):
    metadata_dir = tmp_path / "DISDRODB"

    expected_result = []

    # Test 1 : one metadata file
    key_name = "key1"
    metadata_dict = {key_name: "value1"}
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"

    metadata_filepath = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    expected_result.append(metadata_filepath)
    result = _get_list_all_metadata(
        metadata_dir=str(metadata_dir),
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert expected_result == result

    # Test 2 : two metadata files
    station_name = "station_2"
    metadata_filepath = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    expected_result.append(metadata_filepath)
    result = _get_list_all_metadata(
        metadata_dir=str(metadata_dir),
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert result == expected_result


def test__get_list_metadata_with_data(tmp_path):
    expected_result = []

    base_dir = tmp_path / "data" / "DISDRODB"
    metadata_dir = tmp_path / "metadata" / "DISDRODB"

    # Test 1 : one metadata file + one data file
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"

    key_name = "key1"
    metadata_dict = {key_name: "value1"}
    metadata_filepath = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    _ = create_fake_raw_data_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    expected_result.append(metadata_filepath)

    result = _get_list_metadata_with_data(
        base_dir=base_dir,
        metadata_dir=metadata_dir,
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert result == expected_result

    # Test 1 : two metadata files + one data file
    station_name = "station_2"
    key_name = "key1"
    metadata_dict = {key_name: "value1"}

    metadata_filepath = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    result = _get_list_metadata_with_data(
        base_dir=str(base_dir),
        data_sources=data_source,
        campaign_names=campaign_name,
    )
    assert result == expected_result

    # Test 3 : two metadata files + two data files
    _ = create_fake_raw_data_file(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    expected_result.append(metadata_filepath)

    result = _get_list_metadata_with_data(
        base_dir=str(base_dir),
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert sorted(result) == sorted(expected_result)


def test_get_list_metadata_file(tmp_path):
    metadata_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"
    metadata_filepath = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict={},
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Test 1 : Retrieve specific station name
    result = get_list_metadata(
        metadata_dir=metadata_dir,
        data_sources=data_source,
        campaign_names=campaign_name,
        station_names=station_name,
        with_stations_data=False,
    )
    assert result == [metadata_filepath]

    # Test 2: Retrieve all metadata
    result = get_list_metadata(metadata_dir=metadata_dir, with_stations_data=False)
    assert result == [metadata_filepath]

    # Test 3: Retrieve all metadata with data
    with pytest.raises(ValueError):  # raise error if None
        get_list_metadata(metadata_dir=metadata_dir, with_stations_data=True)

    # Test 4: Check return [] if no metadata
    result = get_list_metadata(metadata_dir=metadata_dir, data_sources="unexisting", with_stations_data=False)
    assert result == []

    result = get_list_metadata(metadata_dir=metadata_dir, station_names="unexisting", with_stations_data=False)
    assert result == []

    result = get_list_metadata(metadata_dir=metadata_dir, campaign_names="unexisting", with_stations_data=False)
    assert result == []

    # Test 5: Check by station names
    result = get_list_metadata(metadata_dir=metadata_dir, station_names=station_name, with_stations_data=False)
    assert [metadata_filepath] == result
