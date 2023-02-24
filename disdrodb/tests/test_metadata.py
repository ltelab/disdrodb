import os
import yaml
from disdrodb.l0 import metadata


PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pytest_files"
)


def test_get_default_metadata():
    assert isinstance(metadata.get_default_metadata_dict(), dict)


def test_write_default_metadata():
    fpath = os.path.join(
        PATH_TEST_FOLDERS_FILES, "test_folders_files_creation", "metadata.yml"
    )

    # create metadata file
    metadata.write_default_metadata(str(fpath))

    # check file exist
    assert os.path.exists(fpath)

    # open it
    with open(str(fpath), "r") as f:
        dictionary = yaml.safe_load(f)

    # check is the expected dictionary
    expected_dict = metadata.get_default_metadata_dict()
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
