import os
import yaml
from disdrodb.l0 import metadata


PATH_TEST_FOLDERS_FILES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytest_files")


def test_get_default_metadata():
    assert isinstance(metadata.get_default_metadata_dict(), dict)


def create_fake_metadata_folder(tmp_path, data_source="data_source", campaign_name="campaign_name"):
    subfolder_path = tmp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)

    assert os.path.exists(subfolder_path)

    return subfolder_path


def test_write_default_metadata(tmp_path):
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "station_name"

    fpath = os.path.join(create_fake_metadata_folder(tmp_path, data_source, campaign_name), f"{station_name}.yml")

    # create metadata file
    metadata.write_default_metadata(str(fpath))

    # check file exist
    assert os.path.exists(fpath)

    # open it
    with open(str(fpath), "r") as f:
        dictionary = yaml.safe_load(f)

    # check is the expected dictionary
    expected_dict = metadata.get_default_metadata_dict()
    expected_dict["data_source"] = data_source
    expected_dict["campaign_name"] = campaign_name
    expected_dict["station_name"] = station_name
    assert expected_dict == dictionary

    # remove dictionary
    if os.path.exists(fpath):
        os.remove(fpath)


def test_read_metadata():
    raw_dir = os.path.join(PATH_TEST_FOLDERS_FILES, "test_folders_files_creation")
    station_name = "123"

    metadata_folder_path = os.path.join(raw_dir, "metadata")

    if not os.path.exists(metadata_folder_path):
        os.makedirs(metadata_folder_path)

    metadata_path = os.path.join(metadata_folder_path, f"{station_name}.yml")

    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    # create data
    data = metadata.get_default_metadata_dict()

    # create metadata file
    metadata.write_default_metadata(str(metadata_path))

    # Read the metadata file
    function_return = metadata.read_metadata(raw_dir, station_name)

    assert function_return == data


def test_check_metadata_compliance():
    # function_return = metadata.check_metadata_compliance()
    # function not implemented
    assert 1 == 1
