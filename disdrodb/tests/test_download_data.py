import os
import yaml

from disdrodb.data_transfer import download_data


def create_fake_config_file(temp_path, data_source, campaign_name, station_name, with_url: bool = True):
    subfolder_path = temp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    subfolder_path.mkdir(parents=True)
    # create a fake yaml file in temp folder
    with open(os.path.join(subfolder_path, f"{station_name}.yml"), "w") as f:
        yaml_dict = {}

        if with_url:
            yaml_dict["data_url"] = "https://www.example.com"
        yaml_dict["station_name"] = "station_name"

        yaml.dump(yaml_dict, f)

    assert os.path.exists(os.path.join(subfolder_path, f"{station_name}.yml"))


def test_get_metadata_folders(tmp_path):
    expected_result = []

    # create on fake config file in temp folder
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name)
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name)
    result = download_data.get_metadata_dirs(testing_path)
    expected_result.append(os.path.join(testing_path, "metadata"))

    assert result == expected_result

    # create a second fake config file in temp folder not is the same data_source folder
    # create on fake config file in temp folder
    data_source = "data_source_2"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name)
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw")
    result = download_data.get_metadata_dirs(testing_path)
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
    result = download_data._get_remote_and_local_data_directories(
        str(os.path.join(tmp_path, "DISDRODB")),
        data_source,
        campaign_name,
        station_name,
    )
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "data", station_name)
    expected_result.append((testing_path, "https://www.example.com"))

    assert expected_result == result

    # test 2 :
    # - one config file with url
    # - one config file without url
    data_source = "data_source2"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name, with_url=False)
    result = download_data._get_remote_and_local_data_directories(str(os.path.join(tmp_path, "DISDRODB")))
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "data", station_name)
    assert expected_result == result

    # test 3 :
    # - 2  config files with url
    # - one config file without url
    data_source = "data_source3"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_config_file(tmp_path, data_source, campaign_name, station_name, with_url=True)
    result = download_data._get_remote_and_local_data_directories(str(os.path.join(tmp_path, "DISDRODB")))
    testing_path = os.path.join(tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "data", station_name)
    expected_result.append((testing_path, "https://www.example.com"))
    assert len(expected_result) == len(result)
