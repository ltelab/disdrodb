import re
import os
import pytest
import uuid

from disdrodb.data_transfer.upload_data import upload_disdrodb_archives
from disdrodb.api.metadata import _read_yaml_file, _write_yaml_file
from disdrodb.utils.zenodo import _create_zenodo_deposition


def create_fake_metadata_file(disdrodb_dir, data_source, campaign_name, station_name, data_url=""):
    metadata_dir = disdrodb_dir / "Raw" / data_source / campaign_name / "metadata"
    if not metadata_dir.exists():
        metadata_dir.mkdir(parents=True)
    metadata_fpath = metadata_dir / f"{station_name}.yml"

    metadata_dict = {}
    if data_url:
        metadata_dict["data_url"] = data_url

    _write_yaml_file(metadata_dict, metadata_fpath)


def create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name):
    data_dir = disdrodb_dir / "Raw" / data_source / campaign_name / "data" / station_name
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    data_fpath = data_dir / "test_data.txt"
    data_fpath.touch()

    return data_dir


def get_metadata_dict(disdrodb_dir, data_source, campaign_name, station_name):
    metadata_fpath = disdrodb_dir / "Raw" / data_source / campaign_name / "metadata" / f"{station_name}.yml"
    return _read_yaml_file(metadata_fpath)


def mock_zenodo_api(requests_mock):
    """Mock Zenodo API."""

    # Deposition creation
    deposition_id = 123456
    bucked_id = str(uuid.uuid4())
    bucket_url = f"https://sandbox.zenodo.org/api/files/{bucked_id}"
    response = {
        "id": deposition_id,
        "links": {"bucket": bucket_url},
    }
    requests_mock.post("https://sandbox.zenodo.org/api/deposit/depositions", json=response, status_code=201)

    # File upload (match any remote file path)
    file_upload_url = re.compile(f"{bucket_url}/.*")
    requests_mock.put(file_upload_url, status_code=200)


def test_upload_to_zenodo(tmp_path, requests_mock):
    """Create two stations. One already has a remote url. Upload to Zenodo sandbox."""

    disdrodb_dir = tmp_path / "DISDRODB"
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"
    station_name1 = "test_station_name1"
    station_name2 = "test_station_name2"
    station_url1 = "https://www.example.com"

    create_fake_metadata_file(disdrodb_dir, data_source, campaign_name, station_name1, station_url1)
    create_fake_metadata_file(disdrodb_dir, data_source, campaign_name, station_name2)
    create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name1)
    create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name2)

    # Set ZENODO_ACCESS_TOKEN environment variable to prevent asking input from user
    os.environ["ZENODO_ACCESS_TOKEN"] = "test_access_token"

    mock_zenodo_api(requests_mock)
    upload_disdrodb_archives(platform="sandbox.zenodo", disdrodb_dir=str(disdrodb_dir))

    # Check metadata files (1st one should not have changed)
    metadata_dict1 = get_metadata_dict(disdrodb_dir, data_source, campaign_name, station_name1)
    new_station_url1 = metadata_dict1["data_url"]
    assert new_station_url1 == station_url1

    metadata_dict2 = get_metadata_dict(disdrodb_dir, data_source, campaign_name, station_name2)
    new_station_url2 = metadata_dict2["data_url"]
    assert new_station_url2.endswith(f"/files/{data_source}/{campaign_name}/{station_name2}.zip")

    # Test upload of already uploaded data
    upload_disdrodb_archives(platform="sandbox.zenodo", disdrodb_dir=str(disdrodb_dir))


def test_wrong_http_response(requests_mock):
    """Test wrong response from Zenodo API."""

    requests_mock.post("https://sandbox.zenodo.org/api/deposit/depositions", json={}, status_code=404)
    with pytest.raises(ValueError):
        _create_zenodo_deposition(sandbox=True)