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
"""Test DISDRODB L0 Directory Creation."""
import os

import pytest 

from disdrodb import __root_path__
from disdrodb.tests.conftest import create_fake_metadata_file
from disdrodb.l0.create_directories import (
    _check_campaign_name_consistency,
    _check_data_source_consistency,
    _copy_station_metadata,
    create_directory_structure,
    create_initial_directory_structure,
    _get_default_metadata_dict,
    create_campaign_default_metadata,
    write_default_metadata,
)
from disdrodb.utils.yaml import read_yaml


TEST_DATA_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data")


def test_create_initial_directory_structure(tmp_path, mocker):
    force = False
    product = "LOA"

    # Define station info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"

    # Define Raw campaign directory structure
    raw_dir = tmp_path / "DISDRODB" / "Raw" / "DATA_SOURCE" / campaign_name
    raw_station_dir = raw_dir / "data" / station_name
    raw_station_dir.mkdir(parents=True)

    # - Add metadata
    metadata_dict = {}
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # - Add fake file
    fake_csv_file_path = os.path.join(raw_station_dir, f"{station_name}.csv")
    with open(fake_csv_file_path, "w") as f:
        f.write("fake csv file")

    # Define Processed campaign directory
    processed_dir = tmp_path / "DISDRODB" / "Processed" / data_source / campaign_name
    processed_dir.mkdir(parents=True)

    # Mock to pass metadata checks
    mocker.patch("disdrodb.metadata.check_metadata.check_metadata_compliance", return_value=None)

    # Execute create_initial_directory_structure
    create_initial_directory_structure(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        station_name=station_name,
        force=force,
        product=product,
    )

    # Test product directory has been created
    expected_folder_path = os.path.join(processed_dir, product)
    assert os.path.exists(expected_folder_path)


# def test_create_initial_directory_structure():
#     campaign_name = "CAMPAIGN_NAME"
#     data_source = "DATA_SOURCE"
#     station_name = "STATION_NAME"
#     product = "L0A"
#     force = True
#     verbose=False

#     raw_dir = os.path.join(
#         TEST_DATA_DIR,
#         "test_dir_structure",
#         "DISDRODB",
#         "Raw",
#         data_source,
#         campaign_name,
#     )
#     processed_dir = os.path.join(
#         TEST_DATA_DIR,
#         "test_dir_creation",
#         "DISDRODB",
#         "Processed",
#         data_source,
#         campaign_name,
#     )
#     # Define expected directory
#     expected_product_dir = os.path.join(processed_dir, product)

#     # TODO:
#     # - Need to remove file to check function works, but then next test is invalidated
#     # - I think we need to create a default directory that we can reinitialize at each test !

#     # Remove directory if exists already
#     if os.path.exists(expected_product_dir):
#         shutil.rmtree(expected_product_dir)
#     assert not os.path.exists(expected_product_dir)

#     # Create directories
#     assert create_directory_structure(processed_dir=processed_dir,
#                                          product=product,
#                                          station_name=station_name,
#                                          force=force,
#                                          verbose=verbose,
#                                          ) is None
#     # Check the directory has been created
#     assert not os.path.exists(expected_product_dir)
#     # TODO:
#     # - check that if data are already present and force=False, raise Error


def test_create_directory_structure(tmp_path, mocker):
    # from pathlib import Path
    # tmp_path = Path("/tmp/test12")
    # tmp_path.mkdir()

    force = False
    product = "L0B"

    # Define station info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"

    # Define Raw campaign directory structure
    raw_dir = tmp_path / "DISDRODB" / "Raw" / "DATA_SOURCE" / campaign_name
    raw_station_dir = raw_dir / "data" / station_name
    raw_station_dir.mkdir(parents=True)

    # - Add metadata
    metadata_dict = {}
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # - Add fake file
    fake_csv_file_path = os.path.join(raw_station_dir, f"{station_name}.csv")
    with open(fake_csv_file_path, "w") as f:
        f.write("fake csv file")

    # Define Processed campaign directory
    processed_dir = tmp_path / "DISDRODB" / "Processed" / data_source / campaign_name
    processed_dir.mkdir(parents=True)

    # subfolder_path = tmp_path / "DISDRODB" / "Processed" / campaign_name / "L0B"
    # subfolder_path.mkdir(parents=True)

    # Mock to pass some internal checks
    mocker.patch("disdrodb.api.io._get_list_stations_with_data", return_value=[station_name])
    mocker.patch("disdrodb.l0.create_directories._check_pre_existing_station_data", return_value=None)

    # Execute create_directory_structure
    create_directory_structure(
        processed_dir=str(processed_dir), product=product, station_name=station_name, force=force, verbose=False
    )

    # Test product directory has been created
    l0a_folder_path = os.path.join(processed_dir, product)
    assert os.path.exists(l0a_folder_path)


# def testcreate_directory_structure():
#     campaign_name = "CAMPAIGN_NAME"
#     data_source = "DATA_SOURCE"
#     station_name = "STATION_NAME"
#     product = "L0B"
#     force = True
#     verbose=False

#     processed_dir = os.path.join(
#         TEST_DATA_DIR,
#         "test_dir_creation",
#         "DISDRODB",
#         "Processed",
#         data_source,
#         campaign_name,
#     )
#     # Define expected directory
#     expected_product_dir = os.path.join(processed_dir, product)

#     # Remove directory if exists already
#     if os.path.exists(expected_product_dir):
#         shutil.rmtree(expected_product_dir)
#     assert not os.path.exists(expected_product_dir)

#     # Create directories
#     assert create_directory_structure(processed_dir=processed_dir,
#                                          product=product,
#                                          station_name=station_name,
#                                          force=force,
#                                          verbose=verbose,
#                                          ) is None
#     # Check the directory has been created
#     assert not os.path.exists(expected_product_dir)
#     # TODO - check that if data are already present and force=False, raise Error


def test_check_campaign_name_consistency():
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    path_raw = os.path.join(
        TEST_DATA_DIR,
        "test_dir_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )
    path_process = os.path.join(
        TEST_DATA_DIR,
        "test_dir_creation",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
    )

    assert _check_campaign_name_consistency(path_raw, path_process) == campaign_name
    
    # Test when is not consistent 
    path_process = os.path.join(
        TEST_DATA_DIR,
        "test_dir_creation",
        "DISDRODB",
        "Processed",
        data_source,
        "ANOTHER_CAMPAIGN_NAME",
    )
    with pytest.raises(ValueError):
        assert _check_campaign_name_consistency(path_raw, path_process)
        

def test_check_data_source_consistency():
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    path_raw = os.path.join(
        TEST_DATA_DIR,
        "test_dir_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )
    
    # Test when consistent 
    path_process = os.path.join(
        TEST_DATA_DIR,
        "test_dir_creation",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
    )

    assert _check_data_source_consistency(path_raw, path_process) == data_source
    
    # Test when is not consistent 
    path_process = os.path.join(
        TEST_DATA_DIR,
        "test_dir_creation",
        "DISDRODB",
        "Processed",
        "ANOTHER_DATA_SOURCE",
        campaign_name,
    )
    with pytest.raises(ValueError):
        assert _check_data_source_consistency(path_raw, path_process)


def test_copy_station_metadata():
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "STATION_NAME"
    raw_dir = os.path.join(
        TEST_DATA_DIR,
        "test_dir_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )
    processed_dir = os.path.join(
        TEST_DATA_DIR,
        "test_dir_creation",
        "DISDRODB",
        "Processed",
        data_source,
        campaign_name,
    )
    destination_metadata_dir = os.path.join(processed_dir, "metadata")

    # Ensure processed_dir and metadata folder exists
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if not os.path.exists(destination_metadata_dir):
        os.makedirs(destination_metadata_dir)

    # Define expected metadata file name
    expected_metadata_fpath = os.path.join(destination_metadata_dir, f"{station_name}.yml")
    # Ensure metadata file does not exist
    if os.path.exists(expected_metadata_fpath):
        os.remove(expected_metadata_fpath)
    assert not os.path.exists(expected_metadata_fpath)

    # Check the function returns None
    assert _copy_station_metadata(raw_dir, processed_dir, station_name) is None

    # Check the function has copied the file
    assert os.path.exists(expected_metadata_fpath)



def create_fake_station_file(
    base_dir, data_source="DATA_SOURCE", campaign_name="CAMPAIGN_NAME", station_name="station_name"
):
    subfolder_path = base_dir / "Raw" / data_source / campaign_name / "data" / station_name
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)

    subfolder_path = base_dir / "Raw" / data_source / campaign_name / "metadata"
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)

    path_file = os.path.join(subfolder_path, f"{station_name}.txt")
    print(path_file)
    with open(path_file, "w") as f:
        f.write("This is some fake text.")


def create_fake_metadata_folder(tmp_path, data_source="DATA_SOURCE", campaign_name="CAMPAIGN_NAME"):
    subfolder_path = tmp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)

    assert os.path.exists(subfolder_path)

    return subfolder_path


def test_create_campaign_default_metadata(tmp_path):
    base_dir = tmp_path / "DISDRODB"
    campaign_name = "test_campaign"
    data_source = "test_data_source"
    station_name = "test_station"

    create_fake_station_file(
        base_dir=base_dir, data_source=data_source, campaign_name=campaign_name, station_name=station_name
    )

    create_campaign_default_metadata(base_dir=base_dir, data_source=data_source, campaign_name=campaign_name)

    expected_file_path = os.path.join(
        tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "metadata", f"{station_name}.yml"
    )

    assert os.path.exists(expected_file_path)


def test_get_default_metadata():
    assert isinstance(_get_default_metadata_dict(), dict)


def test_write_default_metadata(tmp_path):
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    fpath = os.path.join(create_fake_metadata_folder(tmp_path, data_source, campaign_name), f"{station_name}.yml")

    # create metadata file
    write_default_metadata(str(fpath))

    # check file exist
    assert os.path.exists(fpath)

    # open it
    dictionary = read_yaml(str(fpath))

    # check is the expected dictionary
    expected_dict = _get_default_metadata_dict()
    expected_dict["data_source"] = data_source
    expected_dict["campaign_name"] = campaign_name
    expected_dict["station_name"] = station_name
    assert expected_dict == dictionary

    # remove dictionary
    if os.path.exists(fpath):
        os.remove(fpath)
# 