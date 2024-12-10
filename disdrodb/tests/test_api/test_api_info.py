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
"""Test DISDRODB info utility."""

import datetime
import os

import numpy as np
import pytest

from disdrodb.api.info import (
    get_campaign_name_from_filepaths,
    get_end_time_from_filepaths,
    get_info_from_filepath,
    get_key_from_filepath,
    get_key_from_filepaths,
    get_product_from_filepaths,
    get_start_end_time_from_filepaths,
    get_start_time_from_filepaths,
    get_station_name_from_filepaths,
    get_version_from_filepaths,
    infer_base_dir_from_path,
    infer_campaign_name_from_path,
    infer_data_source_from_path,
    infer_disdrodb_tree_path,
    infer_disdrodb_tree_path_components,
    infer_path_info_dict,
)

# Constants for testing
FILE_INFO = {
    "product": "L0A",
    "campaign_name": "LOCARNO_2018",
    "station_name": "60",
    "start_time": "20180625004331",
    "end_time": "20180711010000",
    "version": "1",
    "data_format": "parquet",
}

START_TIME = datetime.datetime.strptime(FILE_INFO["start_time"], "%Y%m%d%H%M%S")
END_TIME = datetime.datetime.strptime(FILE_INFO["end_time"], "%Y%m%d%H%M%S")
VALID_FNAME = (
    "{product:s}.{campaign_name:s}.{station_name:s}.s{start_time:s}.e{end_time:s}.{version:s}.{data_format:s}".format(
        **FILE_INFO,
    )
)
INVALID_FNAME = "invalid_filename.txt"
VALID_KEYS = ["product", "campaign_name", "station_name", "version", "data_format"]
INVALID_KEY = "nonexistent_key"

# valid_filepath = VALID_FNAME


@pytest.fixture
def valid_filepath(tmp_path):
    # Create a valid filepath for testing
    filepath = tmp_path / VALID_FNAME
    filepath.write_text("content does not matter")
    return str(filepath)


@pytest.fixture
def invalid_filepath(tmp_path):
    # Create an invalid filepath for testing
    filepath = tmp_path / INVALID_FNAME
    filepath.write_text("content does not matter")
    return str(filepath)


def test_infer_disdrodb_tree_path_components():
    """Test retrieve correct disdrodb path components."""
    base_dir = os.path.join("whatever_path", "DISDRODB")
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    path_components = [base_dir, "Raw", data_source, campaign_name]
    path = os.path.join(*path_components)
    assert infer_disdrodb_tree_path_components(path) == path_components

    with pytest.raises(ValueError):
        infer_disdrodb_tree_path_components("unvalid_path/because_not_disdrodb")


def test_infer_disdrodb_tree_path():
    # Assert retrieve correct disdrodb path
    disdrodb_path = os.path.join("DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    assert infer_disdrodb_tree_path(path) == disdrodb_path

    # Assert raise error if not disdrodb path
    disdrodb_path = os.path.join("no_disdro_dir", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    with pytest.raises(ValueError):
        infer_disdrodb_tree_path(path)

    # Assert raise error if not valid DISDRODB directory
    disdrodb_path = os.path.join("DISDRODB_UNVALID", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    with pytest.raises(ValueError):
        infer_disdrodb_tree_path(path)

    # Assert it takes the right most DISDRODB occurrence
    disdrodb_path = os.path.join("DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_occurrence", "DISDRODB", "DISDRODB", "directory", disdrodb_path)
    assert infer_disdrodb_tree_path(path) == disdrodb_path

    # Assert behaviour when path == base_dir
    base_dir = os.path.join("home", "DISDRODB")
    assert infer_disdrodb_tree_path(base_dir) == "DISDRODB"


def test_infer_base_dir_from_path():
    # Assert retrieve correct disdrodb path
    base_dir = os.path.join("whatever_path", "is", "before", "DISDRODB")
    disdrodb_path = os.path.join("Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join(base_dir, disdrodb_path)
    assert infer_base_dir_from_path(path) == base_dir

    # Assert raise error if not disdrodb path
    base_dir = os.path.join("whatever_path", "is", "before", "NO_DISDRODB")
    disdrodb_path = os.path.join("Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join(base_dir, disdrodb_path)
    with pytest.raises(ValueError):
        infer_base_dir_from_path(path)

    # Assert behaviour when path == base_dir
    base_dir = os.path.join("home", "DISDRODB")
    assert infer_base_dir_from_path(base_dir) == base_dir


def test_infer_data_source_from_path():
    # Assert retrieve correct
    path = os.path.join("whatever_path", "DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    assert infer_data_source_from_path(path) == "DATA_SOURCE"

    # Assert raise error if path stop at Raw or Processed
    path = os.path.join("whatever_path", "DISDRODB", "Raw")
    with pytest.raises(ValueError):
        infer_data_source_from_path(path)

    # Assert raise error if path not within DISDRODB
    path = os.path.join("whatever_path", "is", "not", "valid")
    with pytest.raises(ValueError):
        infer_data_source_from_path(path)


def test_infer_campaign_name_from_path():
    # Assert retrieve correct
    path = os.path.join("whatever_path", "DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME", "...")
    assert infer_campaign_name_from_path(path) == "CAMPAIGN_NAME"

    # Assert raise error if path stop at Raw or Processed
    path = os.path.join("whatever_path", "DISDRODB", "Raw")
    with pytest.raises(ValueError):
        infer_campaign_name_from_path(path)

    # Assert raise error if path not within DISDRODB
    path = os.path.join("whatever_path", "is", "not", "valid")
    with pytest.raises(ValueError):
        infer_campaign_name_from_path(path)


def test_infer_path_info_dict():
    # Assert retrieve correct
    base_dir = os.path.join("whatever_path", "DISDRODB")
    path = os.path.join(base_dir, "Raw", "DATA_SOURCE", "CAMPAIGN_NAME", "...")
    info_dict = infer_path_info_dict(path)
    assert info_dict["campaign_name"] == "CAMPAIGN_NAME"
    assert info_dict["data_source"] == "DATA_SOURCE"
    assert info_dict["base_dir"] == base_dir

    # Assert raise error if path stop at Raw or Processed
    path = os.path.join("whatever_path", "DISDRODB", "Raw")
    with pytest.raises(ValueError):
        infer_path_info_dict(path)


def test_get_info_from_filepath(valid_filepath):
    # Test if the function correctly parses the file information
    info = get_info_from_filepath(valid_filepath)
    assert info["product"] == FILE_INFO["product"]
    assert info["campaign_name"] == FILE_INFO["campaign_name"]
    assert info["station_name"] == FILE_INFO["station_name"]


def test_get_info_from_filepath_raises_type_error(invalid_filepath):
    # Test if the function raises a TypeError for non-string input
    with pytest.raises(TypeError):
        get_info_from_filepath(1234)  # Intentional bad type


def test_get_info_from_filepath_raises_value_error(invalid_filepath):
    # Test if the function raises a ValueError for an unparsable filename
    with pytest.raises(ValueError):
        get_info_from_filepath(invalid_filepath)


@pytest.mark.parametrize("key", VALID_KEYS)
def test_get_key_from_filepath(valid_filepath, key):
    # Test if the function correctly retrieves the specific key
    product = get_key_from_filepath(valid_filepath, key)
    assert product == FILE_INFO[key]


def test_get_key_from_filepath_raises_key_error(valid_filepath):
    # Test if the function raises a KeyError for a non-existent key
    with pytest.raises(KeyError):
        get_key_from_filepath(valid_filepath, INVALID_KEY)


@pytest.mark.parametrize("key", VALID_KEYS)
def test_get_key_from_filepaths(valid_filepath, key):
    # Test if the function returns a list with the correct keys
    products = get_key_from_filepaths([valid_filepath, valid_filepath], key)
    assert products == [FILE_INFO[key], FILE_INFO[key]]


@pytest.mark.parametrize("key", VALID_KEYS)
def test_get_key_from_filepaths_single_path(valid_filepath, key):
    # Test if the function can handle a single filepath (string) as input
    products = get_key_from_filepaths(valid_filepath, key)
    assert products == [FILE_INFO[key]]


def test_get_version_from_filepath(valid_filepath):
    version = get_version_from_filepaths(valid_filepath)
    assert version == [FILE_INFO["version"]]


def test_get_version_from_filepath_raises_value_error(invalid_filepath):
    with pytest.raises(ValueError):
        get_version_from_filepaths(invalid_filepath)


def test_get_campaign_name_from_filepaths(valid_filepath):
    campaign_name = get_campaign_name_from_filepaths(valid_filepath)
    assert campaign_name == [FILE_INFO["campaign_name"]]


def test_get_station_name_from_filepaths(valid_filepath):
    station_name = get_station_name_from_filepaths(valid_filepath)
    assert station_name == [FILE_INFO["station_name"]]


def test_get_product_from_filepaths(valid_filepath):
    product = get_product_from_filepaths(valid_filepath)
    assert product == [FILE_INFO["product"]]


def test_get_start_time_from_filepaths(valid_filepath):
    start_time = get_start_time_from_filepaths(valid_filepath)
    assert start_time == [START_TIME]


def test_get_end_time_from_filepaths(valid_filepath):
    end_time = get_end_time_from_filepaths(valid_filepath)
    assert end_time == [END_TIME]


def test_get_start_end_time_from_filepaths(valid_filepath):
    start_time, end_time = get_start_end_time_from_filepaths(valid_filepath)
    assert np.array_equal(start_time, np.array([START_TIME]).astype("M8[s]"))
    assert np.array_equal(end_time, np.array([END_TIME]).astype("M8[s]"))
