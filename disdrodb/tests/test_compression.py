import pytest
from disdrodb.utils.compression import compress_station_files


def create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name):
    """Create a station data directory with files inside it.

    station_name
    |-- dir1
        |-- file1.txt
        |-- dir1
            |-- file2.txt
    """

    data_dir = disdrodb_dir / "Raw" / data_source / campaign_name / "data" / station_name
    dir1 = data_dir / "dir1"
    dir2 = dir1 / "dir2"
    if not dir2.exists():
        dir2.mkdir(parents=True)

    file1_txt = dir1 / "file1.txt"
    file1_txt.touch()
    file2_txt = dir2 / "file2.txt"
    file2_txt.touch()


def test_files_compression(tmp_path):
    """Test compression of files in a directory."""

    disdrodb_dir = tmp_path / "DISDRODB"
    data_source = "test_data_source"
    campaign_name = "test_campaign_name"

    methods = ["zip", "gzip", "bzip2"]
    for i in range(len(methods)):
        station_name = f"test_station_name_{i}"
        create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name)
        compress_station_files(disdrodb_dir, data_source, campaign_name, station_name, method=methods[i])

    station_name = "test_station_name"
    create_fake_data_dir(disdrodb_dir, data_source, campaign_name, station_name)
    with pytest.raises(ValueError):
        compress_station_files(disdrodb_dir, data_source, campaign_name, station_name, "unknown_compression_method")
