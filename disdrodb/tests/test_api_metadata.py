import os
import yaml
from disdrodb.api import metadata


def create_fake_metadata_file(
    tmp_path, yaml_file_name, yaml_dict, data_source="data_source", campaign_name="campaign_name"
):
    subfolder_path = tmp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)
    file_path = os.path.join(subfolder_path, yaml_file_name)
    # create a fake yaml file in temp folder
    with open(file_path, "w") as f:
        yaml.dump(yaml_dict, f)

    assert os.path.exists(file_path)

    return file_path


def test_get_metadata_list(tmp_path):
    excepted_result = list()

    # Test 1 : one metadata file
    yaml_file_name = "station_1.yml"
    key_name = "key1"
    yaml_dict = {key_name: "value1"}
    data_source = "data_source"
    campaign_name = "campaign_name"
    fake_metadata_file_path = create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    excepted_result.append(fake_metadata_file_path)
    result = metadata.get_metadata_list(os.path.join(tmp_path, "DISDRODB"))

    assert excepted_result == result

    # Test 2 : two metadata files
    yaml_file_name = "station_2.yml"
    fake_metadata_file_path = create_fake_metadata_file(tmp_path, yaml_file_name, yaml_dict, data_source, campaign_name)
    excepted_result.append(fake_metadata_file_path)
    result = metadata.get_metadata_list(os.path.join(tmp_path, "DISDRODB"))

    assert excepted_result == excepted_result
import os
import yaml

from disdrodb.api.metadata import get_list_metadata


def create_fake_metadata_file(temp_path, data_source, campaign_name, station_name, with_url: bool = True):
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


def test_get_list_metadata_file(tmp_path):
    expected_result = []

    # test 1 :
    # - one config file with url
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_metadata_file(tmp_path, data_source, campaign_name, station_name)
    result = get_list_metadata(str(os.path.join(tmp_path, "DISDRODB")), data_source, campaign_name, station_name, False)
    testing_path = os.path.join(
        tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "metadata", f"{station_name}.yml"
    )
    expected_result.append(testing_path)
    assert expected_result == result

    # test 2 :
    # - downalod_data fucntion without paremeter
    result = get_list_metadata(str(os.path.join(tmp_path, "DISDRODB")), with_stations_data=False)
    assert expected_result == result

    # test 3 :
    # - one config file with url
    # - one config file without url
    data_source = "data_source2"
    campaign_name = "campaign_name"
    station_name = "station_name"
    create_fake_metadata_file(tmp_path, data_source, campaign_name, station_name, with_url=False)
    result = get_list_metadata(str(os.path.join(tmp_path, "DISDRODB")), with_stations_data=False)
    testing_path = os.path.join(
        tmp_path, "DISDRODB", "Raw", data_source, campaign_name, "metadata", f"{station_name}.yml"
    )
    expected_result.append(testing_path)
    assert sorted(expected_result) == sorted(result)
