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
"""Test DISDRODB Download/Upload commands."""

from click.testing import CliRunner

from disdrodb.data_transfer.scripts.disdrodb_upload_station import disdrodb_upload_station
from disdrodb.data_transfer.scripts.disdrodb_upload_archive import disdrodb_upload_archive
from disdrodb.data_transfer.scripts.disdrodb_download_station import disdrodb_download_station
from disdrodb.data_transfer.scripts.disdrodb_download_archive import disdrodb_download_archive
from disdrodb.tests.conftest import create_fake_metadata_file


TEST_ZIP_FPATH = "https://raw.githubusercontent.com/ltelab/disdrodb/main/disdrodb/tests/data/test_data_download/station_files.zip"  # noqa


def test_disdrodb_upload_station(tmp_path):
    """Test the disdrodb_upload_station command."""
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    # - Add fake metadata
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    
    runner = CliRunner()
    runner.invoke(
        disdrodb_upload_station,
        [data_source, campaign_name, station_name, 
         "--base_dir", base_dir],
    )
    

def test_disdrodb_upload_archive(tmp_path):
    """Test the disdrodb_upload_archive command."""
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    # - Add fake metadata
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    
    runner = CliRunner()
    runner.invoke(
        disdrodb_upload_archive,
         ["--base_dir", base_dir],
    )


def test_disdrodb_download_station(tmp_path):
    """Test the disdrodb_download_station command."""
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    # - Add fake metadata
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_dict=metadata_dict,
    )
    
    runner = CliRunner()
    runner.invoke(
        disdrodb_download_station,
        [data_source, campaign_name, station_name, 
         "--base_dir", base_dir],
    )
    

def test_disdrodb_download_archive(tmp_path):
    """Test the disdrodb_download_archive command."""
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"
    
    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    # - Add fake metadata
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_dict=metadata_dict,
    )
        
    runner = CliRunner()
    runner.invoke(
        disdrodb_download_archive,
         ["--base_dir", base_dir],
    )
