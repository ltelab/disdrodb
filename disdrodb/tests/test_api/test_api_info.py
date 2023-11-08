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

import os

import pytest

from disdrodb.api.info import (
    infer_base_dir_from_fpath,
    infer_campaign_name_from_path,
    infer_data_source_from_path,
    infer_disdrodb_tree_path,
    infer_disdrodb_tree_path_components,
)


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


def test_infer_disdrodb_tree_path_components():
    # Assert retrieve correct disdrodb path
    path_components = ["DISDRODB", "Raw", "DATA_SOURCE", "CAMPAIGN_NAME"]
    disdrodb_path = os.path.join(*path_components)
    path = os.path.join("whatever_path", disdrodb_path)
    assert infer_disdrodb_tree_path_components(path) == path_components


def test_infer_base_dir_from_fpath():
    # Assert retrieve correct disdrodb path
    base_dir = os.path.join("whatever_path", "is", "before", "DISDRODB")
    disdrodb_path = os.path.join("Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join(base_dir, disdrodb_path)
    assert infer_base_dir_from_fpath(path) == base_dir

    # Assert raise error if not disdrodb path
    base_dir = os.path.join("whatever_path", "is", "before", "NO_DISDRODB")
    disdrodb_path = os.path.join("Raw", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join(base_dir, disdrodb_path)
    with pytest.raises(ValueError):
        infer_base_dir_from_fpath(path)

    # Assert behaviour when path == base_dir
    base_dir = os.path.join("home", "DISDRODB")
    assert infer_base_dir_from_fpath(base_dir) == base_dir


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
