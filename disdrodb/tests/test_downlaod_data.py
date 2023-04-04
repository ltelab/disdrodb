import os
import pytest

from disdrodb.data_transfer import download_data


def test_check_path():
    # Test a valid path
    path = os.path.abspath(__file__)
    assert download_data.check_path(path) is None

    # Test an invalid path
    path = "/path/that/does/not/exist"
    with pytest.raises(FileNotFoundError):
        download_data.check_path(path)


def test_check_url():
    # Test with valid URLs
    assert download_data.check_url("https://www.example.com") is True
    assert download_data.check_url("http://example.com/path/to/file.html?param=value") is True
    assert download_data.check_url("www.example.com") is True
    assert download_data.check_url("example.com") is True

    # Test with invalid URLs
    assert download_data.check_url("ftp://example.com") is False
    assert download_data.check_url("htp://example.com") is False
    assert download_data.check_url("http://example.com/path with spaces") is False


def create_fake_config_file(temp_path, data_source, campaign_name, station_name, with_url: bool = True):
    subfolder_path = temp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    subfolder_path.mkdir(parents=True)
    # create a fake yaml file in temp folder
    with open(os.path.join(subfolder_path, f"{station_name}.yml"), "w") as f:
        if with_url:
            f.write("data_url: https://www.example.com")
        else:
            f.write("no_url: https://www.example.com")

    assert os.path.exists(os.path.join(subfolder_path, f"{station_name}.yml"))


def test_get_metadata_folders(tmp_path):
    expected_result = []

    # create on fake config file in temp folder
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name)
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name)
    result = download_data.get_metadata_folders(testing_path)
    expected_result.append(os.path.join(testing_path, "metadata"))

    assert result == expected_result

    # create a second fake config file in temp folder not is the same data_source folder
    # create on fake config file in temp folder
    data_source = "data_source_2"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name)
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw")
    result = download_data.get_metadata_folders(testing_path)
    expected_result.append(os.path.join(testing_path, data_source, campaign_name, "metadata"))
    assert sorted(result) == sorted(expected_result)


def test_get_list_urls_local_paths(tmp_path):
    expected_result = []

    # test 1 :
    # - one config file with url
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name)
    result = download_data.get_list_urls_local_paths(
        str(os.path.join(tmp_path, "DISDRODB", "Raw")),
        data_source,
        campaign_name,
        station_name,
    )
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "data", station_name)
    expected_result.append((testing_path, "https://www.example.com"))

    # assert expected_result == result

    # test 2 :
    # - one config file with url
    # - one config file without url
    data_source = "data_source2"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name, with_url=False)
    result = download_data.get_list_urls_local_paths(str(os.path.join(tmp_path, "DISDRODB", "Raw")))
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "data", station_name)
    assert expected_result == result

    # test 3 :
    # - 2  config files with url
    # - one config file without url
    data_source = "data_source3"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name, with_url=True)
    result = download_data.get_list_urls_local_paths(str(os.path.join(tmp_path, "DISDRODB", "Raw")))
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "data", station_name)
    expected_result.append((testing_path, "https://www.example.com"))
    assert len(expected_result) == len(result)
