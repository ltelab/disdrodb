# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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

from disdrodb.cli.disdrodb_download_archive import disdrodb_download_archive
from disdrodb.cli.disdrodb_download_station import disdrodb_download_station
from disdrodb.cli.disdrodb_upload_archive import disdrodb_upload_archive
from disdrodb.cli.disdrodb_upload_station import disdrodb_upload_station
from disdrodb.tests.conftest import create_fake_metadata_file

TEST_ZIP_FPATH = (
    "https://raw.githubusercontent.com/ltelab/disdrodb/main/disdrodb/tests/data/test_data_download/station_files.zip"
)


def test_disdrodb_upload_station(tmp_path):
    """Test the disdrodb_upload_station command."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    # - Add fake metadata
    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    runner = CliRunner()
    runner.invoke(
        disdrodb_upload_station,
        [
            data_source,
            campaign_name,
            station_name,
            "--data_archive_dir",
            data_archive_dir,
            "--metadata_archive_dir",
            metadata_archive_dir,
        ],
    )


def test_disdrodb_upload_archive(tmp_path):
    """Test the disdrodb_upload_archive command."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"

    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    # - Add fake metadata
    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    runner = CliRunner()
    runner.invoke(
        disdrodb_upload_archive,
        ["--data_archive_dir", data_archive_dir, "--metadata_archive_dir", metadata_archive_dir],
    )


def test_disdrodb_download_station(tmp_path):
    """Test the disdrodb_download_station command."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"

    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    # - Add fake metadata
    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_dict=metadata_dict,
    )

    runner = CliRunner()
    runner.invoke(
        disdrodb_download_station,
        [
            data_source,
            campaign_name,
            station_name,
            "--data_archive_dir",
            data_archive_dir,
            "--metadata_archive_dir",
            metadata_archive_dir,
        ],
    )


def test_disdrodb_download_archive(tmp_path):
    """Test the disdrodb_download_archive command."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"

    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = TEST_ZIP_FPATH

    # - Add fake metadata
    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_dict=metadata_dict,
    )

    runner = CliRunner()
    runner.invoke(
        disdrodb_download_archive,
        ["--data_archive_dir", data_archive_dir, "--metadata_archive_dir", metadata_archive_dir],
    )
