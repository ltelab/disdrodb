import os
import yaml
from typing import Dict, List, Set, Union, Optional
from pydantic import BaseModel, ValidationError, validator
import pytest


# Define the pydantic models for *.bins.yaml config files
class L0_data_format_2n_level(BaseModel):
    n_digits: Optional[int]
    n_characters: Optional[int]
    n_decimals: Optional[int]
    data_range: Optional[list]

    @validator("data_range", pre=True)
    def check_list_length(cls, value):
        if value:
            if len(value) != 2:
                raise ValueError("my_list must have exactly 2 elements")
            return value


class L0B_encodings_2n_level(BaseModel):
    dtype: str
    zlib: bool
    complevel: int
    shuffle: bool
    fletcher32: bool
    contiguous: bool
    _FillValue: Optional[Union[int, float]]
    chunksizes: Union[int, list[int]]


# Set paths
ROOT_DISDRODB_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
)
CONFIG_FOLDER = os.path.join(ROOT_DISDRODB_FOLDER, "L0", "configs")


def list_files(path: str, file_name: str) -> List[str]:
    """function that return the list of path files within a directory and subdirectories for file with a given name

    Parameters
    ----------
    path : str
        Path of the directory
    file_name : str
        Name of the file

    Returns
    -------
    List[str]
        list of path files within a directory and subdirectories for file with a given name
    """
    return [
        os.path.join(root, name)
        for root, dirs, files in os.walk(path)
        for name in files
        if name == file_name
    ]


def read_yaml_file(path_file: str) -> Dict:
    """Read a yaml file and return a dictionary.

    Parameters
    ----------
    path_file : str
        path of the yaml file

    Returns
    -------
    Dict
        Content of the yaml file
    """
    with open(path_file, "r") as f:
        try:
            data = yaml.safe_load(f)
        except:
            data = {}

    return data


def validate_schema_pytest(
    schema_to_validate: Union[str, list], schema: BaseModel
) -> bool:
    """function that validate the schema of a given file path with pytest and raise exception if faile to validate

    Parameters
    ----------
    schema_to_validate : Union[str,list]
        Object to validate
    schema : BaseModel
        Base model

    Returns
    -------
    bool
        True is schema correct, False is wrong.
    """

    try:
        model = schema(**schema_to_validate)
        return True
    except:
        return False


def is_dict(obj) -> bool:
    """Fuction to check that a object is a dictionary.

    Parameters
    ----------
    obj : _type_
        Object to check

    Returns
    -------
    bool
        true if object is a dictionary, false if not
    """

    return isinstance(obj, dict)


# Function to check that a object is a list.
def is_list(obj: list) -> bool:
    """Function to check that a object is a list."""
    return isinstance(obj, list)


def is_sorted_int_keys(obj: list) -> bool:
    """Function to check that a list contains only sorted integers keys

    Parameters
    ----------
    obj : list
        List to check

    Returns
    -------
    bool
        True if list contains only sorted integers keys, False if not
    """

    if isinstance(obj, list):
        if len(obj) == 0:
            return True
        else:
            return all(isinstance(x, int) for x in obj) and obj == sorted(obj)
    else:
        return False


def is_numeric_list(obj: list) -> bool:
    """Function to check that a list contains only numeric values

    Parameters
    ----------
    obj : list
        List to check

    Returns
    -------
    bool
        True if list contains only numeric values, False if not
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return True
        else:
            return all(isinstance(x, (int, float)) for x in obj)
    else:
        return False


def is_string_list(obj: list) -> bool:
    """Function to check that a list contains only string values

    Parameters
    ----------
    obj : list
        List to check

    Returns
    -------
    bool
        True if list contains only string values, False if not
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return True
        else:
            return all(isinstance(x, str) for x in obj)
    else:
        return False


list_of_yaml_file_paths = list_files(CONFIG_FOLDER, "L0_data_format.yml")


@pytest.mark.parametrize("yaml_file_path", list_of_yaml_file_paths)
def test_L0_data_format(yaml_file_path: str):
    """Test the format and the content of the L0_data_format.yml file.

    Parameters
    ----------
    yaml_file_path : str
        Path of the yaml file to test
    """
    data = read_yaml_file(yaml_file_path)

    # check that the data is a dictionary
    assert is_dict(data)

    # check that the second level of the dictionary match the schema
    for key, value in data.items():
        assert validate_schema_pytest(value, L0_data_format_2n_level)


# Test *_bins.yaml formatting and content
list_of_files = ["diameter_bins.yml", "velocity_bins.yml"]
list_of_yaml_file_paths = []

for i in list_of_files:
    list_of_yaml_file_paths.extend(list_files(CONFIG_FOLDER, i))


@pytest.mark.parametrize("yaml_file_path", list_of_yaml_file_paths)
def test_bins_format(yaml_file_path: str) -> None:
    """Test the format and the content of *_bins.yml files

    Parameters
    ----------
    yaml_file_path : str
        Path of the yaml file to test
    """
    data = read_yaml_file(yaml_file_path)

    if data:
        # check that the data is a dictionary
        assert is_dict(data)

        # ckeck that the keys are strings
        list_of_fisrt_level_keys = list(data.keys())
        assert is_string_list(list_of_fisrt_level_keys)

        # check that the values are lists
        list_of_fisrt_level_values = list(data.values())
        assert is_list(list_of_fisrt_level_values)

        # check the second level
        for first_level_key, first_level_value in data.items():
            list_of_second_level_keys = list(first_level_value.keys())
            list_of_second_level_values = list(first_level_value.values())

            # check that the keys are sorted integers
            assert is_sorted_int_keys(list_of_second_level_keys)

            # check that the values are numeric
            if first_level_key in ["center", "width"]:
                assert is_numeric_list(list_of_second_level_values)

            if first_level_key in ["bounds"]:
                for second_level_key, second_level_value in first_level_value.items():
                    assert is_numeric_list(second_level_value)

        # check bound and width equls lenght
        assert len(data.get("bounds")) == len(data.get("width"))

        # check that the bounds distance is equal to the width (but not for the first key)
        for id in list(data.get("bounds").keys())[1:-1]:
            [bound_min, bound_max] = data.get("bounds")[id]
            width = data.get("width")[id]
            distance = round(bound_max - bound_min, 3)
            assert distance == width


# Test format and content for basic YAML files
list_of_files = [
    "L0A_encodings.yml",
    "variable_description.yml",
    "variable_longname.yml",
    "variable_units.yml",
    "variables.yml",
]
list_of_yaml_file_paths = []

for i in list_of_files:
    list_of_yaml_file_paths.extend(list_files(CONFIG_FOLDER, i))


@pytest.mark.parametrize("yaml_file_path", list_of_yaml_file_paths)
def test_yaml_format_basic_config_files(yaml_file_path: str) -> None:
    """Test basic YAML file format. It should be a dictionary with string keys and string values.

    Parameters
    ----------
    yaml_file_path : str
        Path of the yaml file to test
    """
    data = read_yaml_file(yaml_file_path)

    # check that the data is a dictionary
    assert is_dict(data)

    # check that the keys are strings
    list_of_fisrt_level_keys = list(data.keys())
    assert is_string_list(list_of_fisrt_level_keys)

    # check the values are strings
    list_of_fisrt_level_values = list(data.values())
    assert is_string_list(list_of_fisrt_level_values)


# Test the fotmat and content of the L0B_encodings.yml file
list_of_yaml_file_paths = list_files(CONFIG_FOLDER, "L0B_encodings.yml")


@pytest.mark.parametrize("yaml_file_path", list_of_yaml_file_paths)
def test_L0B_encodings_format(yaml_file_path: str) -> None:
    """test the L0B_encodings.yml file format

    Parameters
    ----------
    yaml_file_path : str
        Path of the yaml file to test
    """
    data = read_yaml_file(yaml_file_path)

    # check that the data is a dictionary
    assert is_dict(data)

    # ckeck that the keys are strings
    list_of_fisrt_level_keys = list(data.keys())
    assert is_string_list(list_of_fisrt_level_keys)

    # check that the second level of the dictionary match the schema
    for key, value in data.items():
        assert validate_schema_pytest(value, L0B_encodings_2n_level)
