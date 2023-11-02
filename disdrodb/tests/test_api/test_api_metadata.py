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

import os

import yaml

from disdrodb.api.metadata import _get_list_all_metadata, _get_list_metadata_with_data


def create_fake_metadata_file(
    tmp_path, yaml_file_name, yaml_dict, data_source="data_source", campaign_name="campaign_name"
):
    subfolder_path = tmp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)
    file_path = os.path.join(subfolder_path, yaml_file_name)
    # create a fake yaml file in temp folder
    with open(file_path, "w") as f:
        yaml.dump(yaml_dict, f)

    assert os.path.exists(file_path)

    return file_path


def create_fake_data_file(tmp_path, data_source="data_source", campaign_name="campaign_name", station_name=""):
    subfolder_path = tmp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "data" / station_name
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)
    file_path = os.path.join(subfolder_path, "fake_data_file.txt")
    # create a fake yaml file in temp folder
    with open(file_path, "w") as f:
        f.write("This is a fake sample file.")

    assert os.path.exists(file_path)

    return file_path


def test__get_list_all_metadata(tmp_path):
    excepted_result = list()

    # Test 1 : one metadata file
    yaml_file_name = "station_1.yml"
    key_name = "key1"
    yaml_dict = {key_name: "value1"}
    data_source = "data_source"
    campaign_name = "campaign_name"

    fake_metadata_file_path = create_fake_metadata_file(
        tmp_path=tmp_path,
        yaml_file_name=yaml_file_name,
        yaml_dict=yaml_dict,
        data_source=data_source,
        campaign_name=campaign_name,
    )

    excepted_result.append(fake_metadata_file_path)

    result = _get_list_all_metadata(
        disdrodb_dir=os.path.join(tmp_path, "DISDRODB"),
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert excepted_result == result

    # Test 2 : two metadata files
    yaml_file_name = "station_2.yml"
    fake_metadata_file_path = create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    excepted_result.append(fake_metadata_file_path)
    result = _get_list_all_metadata(
        disdrodb_dir=os.path.join(tmp_path, "DISDRODB"),
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert excepted_result == excepted_result


def test_get_list_metadata_with_data(tmp_path):
    expected_result = list()

    # Test 1 : one metadata file + one data file
    station_name = "station_1"
    yaml_file_name = f"{station_name}.yml"
    key_name = "key1"
    yaml_dict = {key_name: "value1"}
    data_source = "data_source"
    campaign_name = "campaign_name"

    fake_metadata_file_path = create_fake_metadata_file(
        tmp_path=tmp_path,
        yaml_file_name=yaml_file_name,
        yaml_dict=yaml_dict,
        data_source=data_source,
        campaign_name=campaign_name,
    )

    expected_result.append(fake_metadata_file_path)

    create_fake_data_file(
        tmp_path=tmp_path, data_source=data_source, campaign_name=campaign_name, station_name=station_name
    )

    result = _get_list_metadata_with_data(
        disdrodb_dir=os.path.join(tmp_path, "DISDRODB"),
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert result == expected_result

    # Test 1 : two metadata files + one data file
    station_name = "station_2"
    yaml_file_name = f"{station_name}.yml"
    key_name = "key1"
    yaml_dict = {key_name: "value1"}

    fake_metadata_file_path = create_fake_metadata_file(
        tmp_path=tmp_path,
        yaml_file_name=yaml_file_name,
        yaml_dict=yaml_dict,
        data_source=data_source,
        campaign_name=campaign_name,
    )

    result = _get_list_metadata_with_data(
        disdrodb_dir=os.path.join(tmp_path, "DISDRODB"),
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert result == expected_result

    # Test 3 : two metadata files + two data files
    create_fake_data_file(
        tmp_path=tmp_path, data_source=data_source, campaign_name=campaign_name, station_name=station_name
    )

    expected_result.append(fake_metadata_file_path)

    result = _get_list_metadata_with_data(
        disdrodb_dir=os.path.join(tmp_path, "DISDRODB"),
        data_sources=data_source,
        campaign_names=campaign_name,
    )

    assert sorted(result) == sorted(expected_result)


#
# def test_get_list_metadata_file(tmp_path):
#     expected_result = []

#     # test 1 :
#     # - one config file with url
#     data_source = "data_source"
#     campaign_name = "campaign_name"
#     station_name = "station_name"
#     create_fake_metadata_file(tmp_path, data_source, campaign_name, station_name)
#     disdrodb_dir = str(os.path.join(tmp_path, "DISDRODB"))
#     result = get_list_metadata(disdrodb_dir, data_source,
#                                campaign_name, station_name, False)
#     testing_path = os.path.join(
#         tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "metadata", f"{station_name}.yml"
#     )
#     expected_result.append(testing_path)
#     assert expected_result == result

#     # test 2 :
#     # - downalod_data function without parameter
#     result = get_list_metadata(str(os.path.join(tmp_path, "DISDRODB")), with_stations_data=False)
#     assert expected_result == result

#     # test 3 :
#     # - one config file with url
#     # - one config file without url
#     data_source = "data_source2"
#     campaign_name = "campaign_name"
#     station_name = "station_name"
#     create_fake_metadata_file(tmp_path, data_source, campaign_name, station_name, with_url=False)
#     result = get_list_metadata(str(os.path.join(tmp_path, "DISDRODB")), with_stations_data=False)
#     testing_path = os.path.join(
#         tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "metadata", f"{station_name}.yml"
#     )
#     expected_result.append(testing_path)
#     assert sorted(expected_result) == sorted(result)
