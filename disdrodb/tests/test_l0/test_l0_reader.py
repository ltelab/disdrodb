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

import pytest

from disdrodb.l0 import l0_reader
from disdrodb.l0.l0_reader import (
    _check_metadata_reader,
    _check_reader_arguments,
    _get_readers_data_sources_path,
    _get_readers_paths_by_data_source,
    available_readers,
    check_available_readers,
    get_reader_function_from_metadata_key,
    get_station_reader_function,
)
from disdrodb.tests.conftest import create_fake_metadata_file
from disdrodb.utils.yaml import read_yaml, write_yaml

# Some test are based on the following reader:
DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "EPFL_2009"


def test_available_readers():
    result = available_readers(data_sources=None, reader_path=False)
    assert isinstance(result, dict)
    assert all(isinstance(value, list) for value in result.values())

    result = available_readers(data_sources="EPFL", reader_path=False)
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
        _check_metadata_reader({"another_key": "whatever"})

    # Test when "reader" key is empty
    with pytest.raises(
        ValueError,
        match="The reader '' reported in the metadata is not valid. Must have '<DATA_SOURCE>/<READER_NAME>' pattern.",
    ):
        _check_metadata_reader({"reader": ""})

    # Test when "reader" key is made of three components
    with pytest.raises(
        ValueError,
        match="Expecting the reader reference to be composed of <DATA_SOURCE>/<READER_NAME>.",
    ):
        _check_metadata_reader({"reader": "ONE/TWO/THREE"})

    # Test when "reader" key is present and valid
    assert _check_metadata_reader({"reader": f"{DATA_SOURCE}/{CAMPAIGN_NAME}"}) is None


def test_check_reader_arguments():
    # Test correct reader
    def good_reader(
        raw_dir,
        processed_dir,
        station_name,
        # Processing options
        force=False,
        verbose=False,
        parallel=False,
        debugging_mode=False,
    ):
        df = "dummy_dataframe"
        return df

    _check_reader_arguments(good_reader)

    # Test bad reader
    def bad_reader(
        bad_arguments,
    ):
        df = "dummy_dataframe"
        return df

    with pytest.raises(ValueError):
        _check_reader_arguments(bad_reader)


def test_get_station_reader_function(tmp_path):
    metadata_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAGIN_NAME"
    station_name = "station_name"

    metadata_dict = {"reader": f"{DATA_SOURCE}/{CAMPAIGN_NAME}"}

    metadata_filepath = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    result = get_station_reader_function(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    assert callable(result)

    # Assert raise error if not reader key in metadata
    metadata_dict = read_yaml(metadata_filepath)
    metadata_dict.pop("reader", None)
    write_yaml(metadata_dict, metadata_filepath)

    with pytest.raises(ValueError, match="The `reader` key is not available in the metadata"):
        get_station_reader_function(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )


def test_get_reader_from_metadata(tmp_path):
    metadata_dir = tmp_path / "DISDRODB"
    data_source = DATA_SOURCE
    campaign_name = CAMPAIGN_NAME
    station_name = "station_name"

    metadata_dict = {"reader": f"{DATA_SOURCE}/{CAMPAIGN_NAME}"}
    reader_data_source_name = f"{DATA_SOURCE}/{CAMPAIGN_NAME}"

    _ = create_fake_metadata_file(
        metadata_dir=metadata_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    result = get_reader_function_from_metadata_key(reader_data_source_name=reader_data_source_name)
    assert callable(result)


def test_get_readers_paths_by_data_source():
    with pytest.raises(ValueError):
        _get_readers_paths_by_data_source(data_source="dummy")


def test_check_available_readers():
    assert check_available_readers() is None


def test_get_reader_function_from_metadata_key():
    reader_data_source_name = f"{DATA_SOURCE}/{CAMPAIGN_NAME}"
    result = get_reader_function_from_metadata_key(reader_data_source_name=reader_data_source_name)
    assert callable(result)


def test__get_readers_data_sources_path():
    result = _get_readers_data_sources_path()
    assert isinstance(result, list)


def test__get_available_readers_dict():
    # Check that at least the EPFL institution is included in the list of readers
    function_return = l0_reader._get_available_readers_dict()
    assert "EPFL" in function_return


def test_check_reader_data_source():
    # Check that at least the EPFL institution is included in the list of readers
    function_return = l0_reader._check_reader_data_source("EPFL")
    assert function_return == "EPFL"

    # Check raise error if not existing data_source
    with pytest.raises(ValueError):
        l0_reader._check_reader_data_source("epfl")

    with pytest.raises(ValueError):
        l0_reader._check_reader_data_source("dummy")


def test__check_reader_exists():
    # Check existing reader
    function_return = l0_reader._check_reader_exists("EPFL", "EPFL_ROOF_2012")
    assert function_return == "EPFL_ROOF_2012"

    # Check unexisting reader
    with pytest.raises(ValueError):
        l0_reader._check_reader_exists("EPFL", "dummy")


def test_get_reader_function():
    # Check that the object is a function
    function_return = l0_reader.get_reader_function("EPFL", "EPFL_ROOF_2012")
    assert inspect.isfunction(function_return)
