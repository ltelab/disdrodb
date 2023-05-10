import os

import pytest
import yaml

from disdrodb.data_transfer import download_data


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


@pytest.mark.parametrize("url", ["https://raw.githubusercontent.com/ltelab/disdrodb/main/README.md"])
def test_download_file_from_url(url, tmp_path):
    download_data._download_file_from_url(url, tmp_path)
    assert os.path.isfile(os.path.join(tmp_path, os.path.basename(url))) is True
