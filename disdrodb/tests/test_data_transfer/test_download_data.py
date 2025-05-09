# #!/usr/bin/env python3

# # -----------------------------------------------------------------------------.
# # Copyright (c) 2021-2023 DISDRODB developers
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program.  If not, see <http://www.gnu.org/licenses/>.
# # -----------------------------------------------------------------------------.
"""Test DISDRODB download utility."""

import os

import pytest

import disdrodb
from disdrodb.api.path import define_station_dir
from disdrodb.data_transfer.download_data import (
    _download_file_from_url,
    _download_station_data,
    download_archive,
    download_station,
)
from disdrodb.tests.conftest import create_fake_metadata_file, create_fake_raw_data_file

TEST_ZIP_FPATH = (
    "https://raw.githubusercontent.com/ltelab/disdrodb/main/disdrodb/tests/data/test_data_download/station_files.zip"
)


def test_download_file_from_url(tmp_path):
    # Test download case when empty directory
    url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/README.md"
    # url = "https://httpbin.org/stream-bytes/1024"
    dst_filepath = _download_file_from_url(url, tmp_path, force=False)
    assert os.path.isfile(dst_filepath)

    # Test download case when directory is not empty and force=False --> avoid download
    url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/CODE_OF_CONDUCT.md"
    # url = "https://httpbin.org/stream-bytes/1025"
    with pytest.raises(ValueError):
        _download_file_from_url(url, tmp_path, force=False)

    # Test download case when directory is not empty and force=True --> it download
    url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/CODE_OF_CONDUCT.md"
    # url = "https://httpbin.org/stream-bytes/1026"
    dst_filepath = _download_file_from_url(url, tmp_path, force=True)
    assert os.path.isfile(dst_filepath)


def test_download_station_data(tmp_path):
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    # Define metadata
    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    # Create metadata file
    metadata_filepath = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_dict=metadata_dict,
    )

    # Download data
    _download_station_data(metadata_filepath=metadata_filepath, data_archive_dir=data_archive_dir, force=True)

    # Define expected station directory
    station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )

    # Assert files in the zip file have been unzipped
    assert os.path.isfile(os.path.join(station_dir, "station_file1.txt"))
    # Assert inner zipped files are not unzipped !
    assert os.path.isfile(os.path.join(station_dir, "station_file2.zip"))
    # Assert inner directories are there
    assert os.path.isdir(os.path.join(station_dir, "2020"))
    # Assert zip file has been removed
    assert not os.path.exists(os.path.join(station_dir, "station_files.zip"))


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("disdrodb_data_url", [None, "", 1])
def test_download_without_any_remote_url(tmp_path, requests_mock, mocker, disdrodb_data_url, force):
    """Test download station data without url."""
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"

    # Create metadata file
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = disdrodb_data_url

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check download station raise error
    with pytest.raises(ValueError):
        download_station(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            force=force,
        )

    # Check download archive run
    download_archive(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_sources=data_source,
        campaign_names=campaign_name,
        station_names=station_name,
        force=force,
    )


def test_download_station_only_with_valid_metadata(tmp_path):
    """Test download of archive stations is not stopped by single stations download errors."""
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"

    # Create metadata file
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["station_name"] = "ANOTHER_STATION_NAME"
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH
    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Test raise error if metadata file is not valid
    with disdrodb.config.set({"metadata_archive_dir": metadata_archive_dir}), pytest.raises(ValueError):
        download_station(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )


@pytest.mark.parametrize("force", [True, False])
def test_download_station(tmp_path, force):
    """Test download station data."""
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"

    # Create metadata file
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Create raw data file
    raw_file_filepath = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    with disdrodb.config.set({"metadata_archive_dir": metadata_archive_dir}):
        # Check download_station raise error if existing data and force=False
        if not force:
            with pytest.raises(ValueError):
                download_station(
                    data_archive_dir=data_archive_dir,
                    data_source=data_source,
                    campaign_name=campaign_name,
                    station_name=station_name,
                    force=force,
                )

            # Check original raw file exists if force=False
            if not force:
                assert os.path.exists(raw_file_filepath)

        # Check download_station overwrite existing files if force=True
        else:
            download_station(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                force=force,
            )
            # Check original raw file does not exist anymore
            if force:
                assert not os.path.exists(raw_file_filepath)


@pytest.mark.parametrize("existing_data", [True, False])
@pytest.mark.parametrize("force", [True, False])
def test_download_archive(tmp_path, force, existing_data):
    """Test download station data."""
    # Define project paths
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_archive_dir = tmp_path / "data" / "DISDRODB"

    # Create metadata file
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create raw data file
    if existing_data:
        raw_file_filepath = create_fake_raw_data_file(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

    # Check download_archive does not raise error if existing data and force=False
    with disdrodb.config.set({"metadata_archive_dir": metadata_archive_dir}):
        download_archive(
            data_archive_dir=data_archive_dir,
            data_sources=data_source,
            campaign_names=campaign_name,
            station_names=station_name,
            force=force,
        )

    # Check existing_data
    if existing_data:
        if not force:
            # Check original raw file exists if force=False
            assert os.path.exists(raw_file_filepath)
        else:
            # Check original raw file does not exist anymore if force=True
            assert not os.path.exists(raw_file_filepath)
