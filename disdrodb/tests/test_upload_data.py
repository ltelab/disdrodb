from disdrodb.data_transfer.upload_data import upload_disdrodb_archives
from disdrodb.api.metadata import _read_yaml_file, _write_yaml_file


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


def get_metadata_dict(disdrodb_dir, data_source, campaign_name, station_name):
    metadata_fpath = disdrodb_dir / "Raw" / data_source / campaign_name / "metadata" / f"{station_name}.yml"
    return _read_yaml_file(metadata_fpath)


def test_upload_to_zenodo(tmp_path):
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

    # Must set ZENODO_ACCESS_TOKEN environment variable in GitHub
    upload_disdrodb_archives(platform="sandbox.zenodo", files_compression="gzip", disdrodb_dir=str(disdrodb_dir))

    # Check metadata files (1st one should not have changed)
    metadata_dict1 = get_metadata_dict(disdrodb_dir, data_source, campaign_name, station_name1)
    new_station_url1 = metadata_dict1["data_url"]
    assert new_station_url1 == station_url1

    metadata_dict2 = get_metadata_dict(disdrodb_dir, data_source, campaign_name, station_name2)
    new_station_url2 = metadata_dict2["data_url"]
    assert new_station_url2.endswith(f"/files/{data_source}/{campaign_name}/{station_name2}.zip")
