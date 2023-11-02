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
"""Test DISDRODB L0 metadata routines."""

import os

import yaml

from disdrodb import __root_path__
from disdrodb.l0.metadata import (
    create_campaign_default_metadata,
    get_default_metadata_dict,
    read_metadata,
    write_default_metadata,
)

PATH_TEST_FOLDERS_FILES = os.path.join(__root_path__, "disdrodb", "tests", "data")


def create_fake_station_file(
    base_dir, data_source="data_source", campaign_name="campaign_name", station_name="station_name"
):
    subfolder_path = base_dir / "DISDRODB" / "Raw" / data_source / campaign_name / "data" / station_name
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)

    subfolder_path = base_dir / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)

    path_file = os.path.join(subfolder_path, f"{station_name}.txt")
    print(path_file)
    with open(path_file, "w") as f:
        f.write("This is some fake text.")


def test_create_campaign_default_metadata(tmp_path):
    base_dir = os.path.join(tmp_path, "DISDRODB")
    campaign_name = "test_campaign"
    data_source = "test_data_source"
    station_name = "test_station"

    create_fake_station_file(
        base_dir=tmp_path, data_source=data_source, campaign_name=campaign_name, station_name=station_name
    )

    create_campaign_default_metadata(base_dir=base_dir, data_source=data_source, campaign_name=campaign_name)

    expected_file_path = os.path.join(
        tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "metadata", f"{station_name}.yml"
    )

    assert os.path.exists(expected_file_path)


def test_get_default_metadata():
    assert isinstance(get_default_metadata_dict(), dict)


def create_fake_metadata_folder(tmp_path, data_source="data_source", campaign_name="campaign_name"):
    subfolder_path = tmp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)

    assert os.path.exists(subfolder_path)

    return subfolder_path


def test_write_default_metadata(tmp_path):
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name"

    fpath = os.path.join(create_fake_metadata_folder(tmp_path, data_source, campaign_name), f"{station_name}.yml")

    # create metadata file
    write_default_metadata(str(fpath))

    # check file exist
    assert os.path.exists(fpath)

    # open it
    with open(str(fpath)) as f:
        dictionary = yaml.safe_load(f)

    # check is the expected dictionary
    expected_dict = get_default_metadata_dict()
    expected_dict["data_source"] = data_source
    expected_dict["campaign_name"] = campaign_name
    expected_dict["station_name"] = station_name
    assert expected_dict == dictionary

    # remove dictionary
    if os.path.exists(fpath):
        os.remove(fpath)


def test_read_metadata():
    raw_dir = os.path.join(PATH_TEST_FOLDERS_FILES, "test_folders_files_creation")
    station_name = "123"

    metadata_folder_path = os.path.join(raw_dir, "metadata")

    if not os.path.exists(metadata_folder_path):
        os.makedirs(metadata_folder_path)

    metadata_path = os.path.join(metadata_folder_path, f"{station_name}.yml")

    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    # create data
    data = get_default_metadata_dict()

    # create metadata file
    write_default_metadata(str(metadata_path))

    # Read the metadata file
    function_return = read_metadata(raw_dir, station_name)

    assert function_return == data


def test_check_metadata_compliance():
    # function_return = metadata.check_metadata_compliance()
    # function not implemented
    assert 1 == 1
