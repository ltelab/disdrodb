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
"""Test DISDRODB upload utility."""


import re
import uuid

import pytest

import disdrodb
from disdrodb.data_transfer.upload_data import upload_archive, upload_station
from disdrodb.data_transfer.zenodo import _create_zenodo_deposition
from disdrodb.metadata import read_station_metadata
from disdrodb.tests.conftest import create_fake_metadata_file, create_fake_raw_data_file

DEPOSIT_ID = 123456


def test_wrong_http_response(requests_mock):
    """Test wrong response from Zenodo API."""
    requests_mock.post("https://sandbox.zenodo.org/api/deposit/depositions", json={}, status_code=404)
    with pytest.raises(ValueError):
        _create_zenodo_deposition(sandbox=True)


def mock_zenodo_api(requests_mock, zenodo_host):
    """Mock Zenodo API."""
    deposit_id = DEPOSIT_ID
    bucked_id = str(uuid.uuid4())
    bucket_url = f"https://{zenodo_host}.org/api/files/{bucked_id}"
    deposit_url = f"https://{zenodo_host}.org/api/deposit/depositions"
    deposit_id_url = f"https://{zenodo_host}.org/api/deposit/depositions/{deposit_id}"
    response = {
        "id": deposit_id,
        "links": {"bucket": bucket_url},
    }
    # Deposition creation
    requests_mock.post(deposit_url, json=response, status_code=201)

    # File upload (match any remote file path)
    file_upload_url = re.compile(f"{bucket_url}/.*")
    requests_mock.put(file_upload_url, status_code=201)

    # Metadata update
    requests_mock.put(deposit_id_url, json=response, status_code=200)


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("station_url", ["existing_url", ""])
@pytest.mark.parametrize("platform", ["sandbox.zenodo", "zenodo"])
def test_upload_station(tmp_path, requests_mock, mocker, station_url, force, platform):
    """Test upload of station data."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = station_url

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    _ = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Define token name
    token_key = "zenodo_sandbox_token" if platform == "sandbox.zenodo" else "zenodo_token"

    with disdrodb.config.set({token_key: "test_access_token"}):
        mock_zenodo_api(requests_mock, zenodo_host=platform)
        mocker.patch("disdrodb.data_transfer.zenodo._define_disdrodb_data_url", return_value="dummy_url")

        if station_url == "existing_url" and not force:
            with pytest.raises(ValueError):
                upload_station(
                    platform=platform,
                    data_archive_dir=data_archive_dir,
                    metadata_archive_dir=metadata_archive_dir,
                    data_source=data_source,
                    campaign_name=campaign_name,
                    station_name=station_name,
                    force=force,
                )
        else:
            upload_station(
                platform=platform,
                data_archive_dir=data_archive_dir,
                metadata_archive_dir=metadata_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                force=force,
            )

            # Check metadata has changed
            metadata_dict = read_station_metadata(
                metadata_archive_dir=metadata_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
            )
            new_station_url = metadata_dict["disdrodb_data_url"]
            assert new_station_url == "dummy_url"


def test_upload_with_invalid_platform(tmp_path, requests_mock, mocker):
    """Test upload of station data."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"

    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    force = True
    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = "existing_url"

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    _ = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check it raise error if invalid platform
    with pytest.raises(NotImplementedError):
        upload_station(
            platform="invalid_platform",
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            force=force,
        )

    with pytest.raises(NotImplementedError):
        upload_archive(
            platform="invalid_platform",
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_sources=data_source,
            campaign_names=campaign_name,
            station_names=station_name,
            force=force,
        )


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("station_url", ["existing_url", ""])
@pytest.mark.parametrize("platform", ["sandbox.zenodo", "zenodo"])
def test_upload_archive(tmp_path, requests_mock, mocker, station_url, force, platform):
    """Test upload of archive stations data."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = station_url

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    _ = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Define token name
    token_key = "zenodo_sandbox_token" if platform == "sandbox.zenodo" else "zenodo_token"

    # Upload data
    with disdrodb.config.set({token_key: "test_access_token"}):
        mock_zenodo_api(requests_mock, zenodo_host=platform)
        mocker.patch("disdrodb.data_transfer.zenodo._define_disdrodb_data_url", return_value="dummy_url")
        upload_archive(
            platform=platform,
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_sources=data_source,
            campaign_names=campaign_name,
            station_names=station_name,
            force=force,
        )

    # Check metadata has changed
    metadata_dict = read_station_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    new_station_url = metadata_dict["disdrodb_data_url"]
    if station_url != "" and not force:
        # No upload --> No key update
        assert new_station_url == station_url
    else:
        # Upload --> key update
        assert new_station_url == "dummy_url"


@pytest.mark.parametrize("platform", ["sandbox.zenodo", "zenodo"])
def test_upload_archive_do_not_stop(tmp_path, requests_mock, mocker, platform):
    """Test upload of archive stations is not stopped by station errors."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = "dummy_url"

    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    _ = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    mocker.patch("disdrodb.data_transfer.upload_data.upload_station", side_effect=Exception("Whatever error occurred"))
    upload_archive(
        platform=platform,
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_sources=data_source,
        campaign_names=campaign_name,
        station_names=station_name,
        force=False,
    )
