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
from disdrodb.tests.conftest import create_fake_metadata_file


def create_fake_data_dir(
    base_dir, data_source="DATA_SOURCE", campaign_name="CAMPAIGN_NAME", station_name="station_name"
):
    data_dir = base_dir / "Raw" / data_source / campaign_name / "data" / station_name
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    # Create fake file
    data_fpath = data_dir / "test_data.txt"
    data_fpath.touch()

    return data_dir


def test_wrong_http_response(requests_mock):
    """Test wrong response from Zenodo API."""

    requests_mock.post("https://sandbox.zenodo.org/api/deposit/depositions", json={}, status_code=404)
    with pytest.raises(ValueError):
        _create_zenodo_deposition(sandbox=True)


def mock_zenodo_api(requests_mock):
    """Mock Zenodo API."""
    deposit_id = 123456
    bucked_id = str(uuid.uuid4())
    bucket_url = f"https://sandbox.zenodo.org/api/files/{bucked_id}"
    deposit_url = f"https://sandbox.zenodo.org/api/deposit/depositions/{deposit_id}"
    response = {
        "id": deposit_id,
        "links": {"bucket": bucket_url},
    }
    # Deposition creation
    requests_mock.post("https://sandbox.zenodo.org/api/deposit/depositions", json=response, status_code=201)

    # File upload (match any remote file path)
    file_upload_url = re.compile(f"{bucket_url}/.*")
    requests_mock.put(file_upload_url, status_code=201)

    # Metadata update
    requests_mock.put(deposit_url, json=response, status_code=200)


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("station_url", ["https://www.example.com", ""])
def test_station_upload(tmp_path, requests_mock, station_url, force):
    """Test upload of already uploaded data (force=True)."""
    base_dir = tmp_path / "DISDRODB"
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = station_url

    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    create_fake_data_dir(
        base_dir=base_dir, data_source=data_source, campaign_name=campaign_name, station_name=station_name
    )

    # Upload data (with force=False)
    with disdrodb.config.set({"zenodo_sandbox_token": "test_access_token"}):
        mock_zenodo_api(requests_mock)
        upload_archive(
            platform="zenodo.sandbox",
            base_dir=str(base_dir),
            data_sources=data_source,
            campaign_names=campaign_name,
            station_names=station_name,
            force=force,
        )

    # Check metadata has changed
    metadata_dict = read_station_metadata(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    new_station_url = metadata_dict["disdrodb_data_url"]
    if force:
        # Check key update
        assert new_station_url != station_url
    else:
        if station_url == "":
            # upload --> key update
            assert new_station_url != station_url
        else:
            # no upload --> no key update
            assert new_station_url == station_url


def test_station_upload_raise_error(tmp_path, requests_mock):
    """Test upload of already uploaded data (force=True)."""
    base_dir = tmp_path / "DISDRODB"
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name = "test_station_name"

    metadata_dict = {}
    metadata_dict["disdrodb_data_url"] = "existing_url"

    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    create_fake_data_dir(
        base_dir=base_dir, data_source=data_source, campaign_name=campaign_name, station_name=station_name
    )

    # Upload data (with force=False)
    with disdrodb.config.set({"zenodo_sandbox_token": "test_access_token"}):
        mock_zenodo_api(requests_mock)
        with pytest.raises(ValueError):
            upload_station(
                platform="zenodo.sandbox",
                base_dir=str(base_dir),
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                force=False,
            )
