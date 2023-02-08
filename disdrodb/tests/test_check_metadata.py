import os
import pytest
import yaml


from disdrodb.L0 import check_metadata


PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pytest_files"
)


def test_read_yaml():
    # test based on files under tests\pytest_files\test_check_metadata

    # Test reading a valid YAML file
    valid_yaml_attrs = {"key1": "value1", "key2": "value2"}
    yaml_temp_path = os.path.join(
        PATH_TEST_FOLDERS_FILES, "test_check_metadata", "valid.yaml"
    )
    assert check_metadata.read_yaml(yaml_temp_path) == valid_yaml_attrs

    # Test reading a non-existent YAML file
    yaml_temp_path_non_existent = os.path.join(
        PATH_TEST_FOLDERS_FILES, "test_check_metadata", "non_existent.yaml"
    )
    with pytest.raises(FileNotFoundError):
        check_metadata.read_yaml(yaml_temp_path_non_existent)

    # Test reading a YAML file with invalid syntax
    yaml_temp_path_nvalid = os.path.join(
        PATH_TEST_FOLDERS_FILES, "test_check_metadata", "invalid.yaml"
    )
    with pytest.raises(yaml.YAMLError):
        check_metadata.read_yaml(yaml_temp_path_nvalid)


def test_identify_empty_metadata_keys():
    yaml_temp_path = [
        os.path.join(PATH_TEST_FOLDERS_FILES, "test_check_metadata", "valid.yaml")
    ]

    keys = ["key1"]
    function_return = check_metadata.identify_empty_metadata_keys(yaml_temp_path, keys)
    assert function_return is None


def test_identify_missing_metadata_coords():
    yaml_temp_path_valid = [
        os.path.join(
            PATH_TEST_FOLDERS_FILES, "test_check_metadata", "valid_coords.yaml"
        )
    ]

    function_return = check_metadata.identify_missing_metadata_coords(
        yaml_temp_path_valid
    )
    assert function_return is None

    yaml_temp_path_invalid = [
        os.path.join(
            PATH_TEST_FOLDERS_FILES, "test_check_metadata", "invalid_coords.yaml"
        )
    ]

    with pytest.raises(TypeError):
        check_metadata.identify_missing_metadata_coords(yaml_temp_path_invalid)
