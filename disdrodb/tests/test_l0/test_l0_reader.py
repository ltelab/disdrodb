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
"""Test DISDRODB L0 readers routines."""

import inspect
import os

import pytest

from disdrodb.l0 import l0_reader
from disdrodb.l0.l0_reader import (
    _check_metadata_reader,
    _get_readers_data_sources_path,
    _get_readers_paths_by_data_source,
    available_readers,
    check_available_readers,
    get_reader_from_metadata_reader_key,
    get_station_reader_function,
)
from disdrodb.utils.yaml import write_yaml

# Some test are based on the following reader:
DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "EPFL_2009"


def create_fake_metadata_file(
    tmp_path, yaml_file_name, yaml_dict, data_source="data_source", campaign_name="campaign_name"
):
    subfolder_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "metadata")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path, exist_ok=True)
    file_path = os.path.join(subfolder_path, yaml_file_name)
    # create a fake yaml file in temp folder
    write_yaml(yaml_dict, file_path)
    assert os.path.exists(file_path)
    return file_path


def test_available_readers():
    result = available_readers(data_sources=None, reader_path=False)
    assert isinstance(result, dict)
    assert all(isinstance(value, list) for value in result.values())


def test_check_metadata_reader():
    # Test when "reader" key is missing
    with pytest.raises(ValueError, match="The reader is not specified in the metadata."):
        _check_metadata_reader({})

    # Test when "reader" key is present but invalid
    with pytest.raises(ValueError, match="The reader 'invalid_reader' reported in the metadata is not valid."):
        _check_metadata_reader({"reader": "invalid_reader"})

    # Test when "reader" key is not present
    with pytest.raises(ValueError, match="The reader is not specified in the metadata."):
        _check_metadata_reader({"reader2": f"{DATA_SOURCE}/{CAMPAIGN_NAME}"})

    # Test when "reader" key is present and valid
    assert _check_metadata_reader({"reader": f"{DATA_SOURCE}/{CAMPAIGN_NAME}"}) is None


def test_get_station_reader_function(tmp_path):
    base_dir = os.path.join(tmp_path, "DISDRODB")
    station_name = "station_1"
    yaml_dict = {"reader": f"{DATA_SOURCE}/{CAMPAIGN_NAME}"}
    data_source = "data_source"
    campaign_name = "campaign_name"

    create_fake_metadata_file(
        tmp_path=tmp_path,
        yaml_file_name=f"{station_name}.yml",
        yaml_dict=yaml_dict,
        data_source=data_source,
        campaign_name=campaign_name,
    )

    result = get_station_reader_function(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    assert callable(result)


def test_get_reader_from_metadata(tmp_path):
    station_name = "station_1"
    yaml_dict = {"reader": f"{DATA_SOURCE}/{CAMPAIGN_NAME}"}
    data_source = DATA_SOURCE
    campaign_name = CAMPAIGN_NAME
    reader_data_source_name = f"{DATA_SOURCE}/{CAMPAIGN_NAME}"

    create_fake_metadata_file(
        tmp_path=tmp_path,
        yaml_file_name=f"{station_name}.yml",
        yaml_dict=yaml_dict,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    result = get_reader_from_metadata_reader_key(reader_data_source_name=reader_data_source_name)
    assert callable(result)


def test_get_readers_paths_by_data_source():
    with pytest.raises(ValueError):
        _get_readers_paths_by_data_source(data_source="dummy")


def test_check_available_readers():
    assert check_available_readers() is None


def test_get_reader_from_metadata_reader_key():
    reader_data_source_name = f"{DATA_SOURCE}/{CAMPAIGN_NAME}"
    result = get_reader_from_metadata_reader_key(reader_data_source_name=reader_data_source_name)
    assert callable(result)


def test__get_readers_data_sources_path():
    result = _get_readers_data_sources_path()
    assert isinstance(result, list)


def test_get_available_readers_dict():
    # Check that at least the EPFL institution is included in the list of readers
    function_return = l0_reader.get_available_readers_dict()
    assert "EPFL" in function_return.keys()


def test_check_reader_data_source():
    # Check that at least the EPFL institution is included in the list of readers
    function_return = l0_reader._check_reader_data_source("EPFL")
    assert function_return == "EPFL"

    # Check raise error if not existing data_source
    with pytest.raises(ValueError):
        l0_reader._check_reader_data_source("epfl")

    with pytest.raises(ValueError):
        l0_reader._check_reader_data_source("dummy")


def test_check_reader_exists():
    # Check existing reader
    function_return = l0_reader.check_reader_exists("EPFL", "EPFL_ROOF_2012")
    assert function_return == "EPFL_ROOF_2012"

    # Check unexisting reader
    with pytest.raises(ValueError):
        l0_reader.check_reader_exists("EPFL", "dummy")


def test_get_reader():
    # Check that the object is a function
    function_return = l0_reader.get_reader("EPFL", "EPFL_ROOF_2012")
    assert inspect.isfunction(function_return)
