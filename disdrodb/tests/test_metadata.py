import os
import yaml
from disdrodb.L0 import metadata


PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pytest_files"
)


def test_create_metadata():

    fpath = os.path.join(
        PATH_TEST_FOLDERS_FILES, "test_folders_files_creation", "metadata.yml"
    )

    data = metadata.get_attrs_standards()

    # create metadata file
    metadata.create_metadata(str(fpath))

    assert os.path.exists(fpath)

    with open(str(fpath), "r") as f:
        data = yaml.safe_load(f)

    if os.path.exists(fpath):
        os.remove(fpath)


def test_read_metadata():

    raw_dir = os.path.join(PATH_TEST_FOLDERS_FILES, "test_folders_files_creation")
    station_id = "123"

    metadata_folder_path = os.path.join(raw_dir, "metadata")

    if not os.path.exists(metadata_folder_path):
        os.makedirs(metadata_folder_path)

    metadata_path = os.path.join(metadata_folder_path, f"{station_id}.yml")

    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    # create data
    data = metadata.get_attrs_standards()

    # create metadata file
    metadata.create_metadata(str(metadata_path))

    # Read the metadata file
    function_return = metadata.read_metadata(raw_dir, station_id)

    assert function_return == data


def test_check_metadata_compliance():
    # function_return = metadata.check_metadata_compliance()
    # function not implemented
    assert 1 == 1
