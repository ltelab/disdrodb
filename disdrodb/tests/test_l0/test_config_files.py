#!/usr/bin/env python3

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2023 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Check DISDRODB L0 configuration files."""
import os
from typing import Union

import pytest
import yaml
from pydantic import BaseModel

from disdrodb import package_dir
from disdrodb.utils.directories import list_files

CONFIG_FOLDER = os.path.join(package_dir, "l0", "configs")


def read_yaml_file(filepath: str) -> dict:
    """Read a YAML file and return a dictionary.

    Parameters
    ----------
    filepath : str
        Path of the YAML file

    Returns
    -------
    dictionary: dict
        Content of the YAML file
    """
    with open(filepath) as f:
        try:
            data = yaml.safe_load(f)
        except Exception:
            data = {}
    return data


def is_dict(obj) -> bool:
    """Check that a object is a dictionary.

    Parameters
    ----------
    obj : _type_
        Object to check.

    Returns
    -------
    result: bool
        True if object is a dictionary, False if not.
    """
    return isinstance(obj, dict)


def is_list(obj: list) -> bool:
    """Function to check that a object is a list."""
    return isinstance(obj, list)


def is_sorted_int_keys(obj: list) -> bool:
    """Check that a list contains only sorted integers keys.

    Parameters
    ----------
    obj : list
        List to check.

    Returns
    -------
    result: bool
        True if list contains only sorted integers keys, False if not.
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return True
        return all(isinstance(x, int) for x in obj) and obj == sorted(obj)
    return False


def is_numeric_list(obj: list) -> bool:
    """Check that a list contains only numeric values.

    Parameters
    ----------
    obj : list
        List to check

    Returns
    -------
    result: bool
        True if list contains only numeric values, False if not.
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return True
        return all(isinstance(x, (int, float)) for x in obj)
    return False


def is_string_list(obj: list) -> bool:
    """Check that a list contains only string values.

    Parameters
    ----------
    obj : list
        List to check.

    Returns
    -------
    result: bool
        True if list contains only string values, False if not.
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return True
        return all(isinstance(x, str) for x in obj)
    return False


def validate_schema_pytest(schema_to_validate: Union[str, list], schema: BaseModel) -> bool:
    """Validate the schema of a given file path with pytest.

    It raise an Exception if failed to validate.

    Parameters
    ----------
    schema_to_validate : Union[str,list]
        Object to validate
    schema : BaseModel
        Base model

    Returns
    -------
    result: bool
        True is schema correct, False is wrong.
    """
    try:
        schema(**schema_to_validate)
        return True
    except Exception:
        return False


class L0BVariableAttributesSchema(BaseModel):
    """Define the expected keys and values of the each variable in the l0b_cf_attrs file."""

    description: str
    units: str
    long_name: str


# Test the format and content of the l0b_cf_attrs.yml files
yaml_filepaths = list_files(CONFIG_FOLDER, "l0b_cf_attrs.yml", recursive=True)


@pytest.mark.parametrize("yaml_filepath", yaml_filepaths)
def test_l0b_cf_attrs_format(yaml_filepath: str) -> None:
    """Test the l0b_cf_attrs.yml file format.

    Parameters
    ----------
    yaml_filepath : str
        Path of the l0b_cf_attrs.yml file to test.
    """
    data = read_yaml_file(yaml_filepath)
    assert is_dict(data)
    assert is_string_list(list(data.keys()))
    # Check the second level of the dictionary match the schema
    for value in data.values():
        assert validate_schema_pytest(value, L0BVariableAttributesSchema)


# Test the format and content of the *_bins.yml files
filenames = ["bins_diameter.yml", "bins_velocity.yml"]
yaml_filepaths = []
for filename in filenames:
    yaml_filepaths.extend(list_files(CONFIG_FOLDER, filename, recursive=True))


@pytest.mark.parametrize("yaml_filepath", yaml_filepaths)
def test_bins_format(yaml_filepath: str) -> None:
    """Test the bins_*.yml file format.

    Parameters
    ----------
    yaml_filepath : str
        Path of the YAML file to test.
    """
    data = read_yaml_file(yaml_filepath)
    if data:  # deal with empty bins_velocity.yml (impact disdrometers and ODM-470)
        assert is_dict(data)
        assert is_string_list(list(data.keys()))
        assert is_list(list(data.values()))
        # Check the second level of the dictionary
        for first_level_key, first_level_value in data.items():
            list_of_nested_keys = list(first_level_value.keys())
            list_of_nested_values = list(first_level_value.values())
            # Check that the keys are sorted integers
            assert is_sorted_int_keys(list_of_nested_keys)
            # Check that the values are numeric
            if first_level_key in ["center", "width"]:
                assert is_numeric_list(list_of_nested_values)
            # Check that 'bounds' is a list
            if first_level_key in ["bounds"]:
                for second_level_value in first_level_value.values():
                    assert is_numeric_list(second_level_value)

        # Check bound and width equals length
        assert len(data.get("bounds")) == len(data.get("width"))

        # Check that the bounds distance is equal to the width (but not for the first key)
        for idx in list(data.get("bounds").keys())[1:-1]:
            [bound_min, bound_max] = data.get("bounds")[idx]
            width = round(data.get("width")[idx], 3)
            distance = round(bound_max - bound_min, 3)
            assert distance == width


####---------------------------------------------------------------------------.
#### Deprecated
# - Has been moved to check.configs and test_check_configs


# class RawDataFormatSchema(BaseModel):
#     """Define the expected keys and values of the each variable in the raw_data_format.yml file."""

#     n_digits: Optional[int]
#     n_characters: Optional[int]
#     n_decimals: Optional[int]
#     data_range: Optional[list]

#     @field_validator("data_range")
#     def check_list_length(cls, value):
#         if value:
#             if len(value) != 2:
#                 raise ValueError("data_range must have exactly 2 elements")
#             return value


# # Test the format and content of the raw_data_format.yml files
# yaml_filepaths = list_files(CONFIG_FOLDER, "raw_data_format.yml", recursive=True)


# @pytest.mark.parametrize("yaml_filepath", yaml_filepaths)
# def test_raw_data_format(yaml_filepath: str):
#     """Test the raw_data_format.yml file format.

#     Parameters
#     ----------
#     yaml_filepath : str
#         Path of the raw_data_format.yml file to test.
#     """
#     data = read_yaml_file(yaml_filepath)
#     assert is_dict(data)
#     assert is_string_list(list(data.keys()))
#     # Check the second level of the dictionary match the schema
#     for value in data.values():
#         assert validate_schema_pytest(value, RawDataFormatSchema)


# # Test format and content of l0a_encodings.yml
# yaml_filepaths = list_files(CONFIG_FOLDER, "l0a_encodings.yml", recursive=True)


# @pytest.mark.parametrize("yaml_filepath", yaml_filepaths)
# def test_l0a_encodings_format(yaml_filepath: str) -> None:
#     """Test the l0a_encoding.yml format.

#     It should be a dictionary with string keys and string values.

#     Parameters
#     ----------
#     yaml_filepath : str
#         Path of the yaml file to test.
#     """
#     data = read_yaml_file(yaml_filepath)
#     assert is_dict(data)
#     assert is_string_list(list(data.keys()))
#     assert is_string_list(list(data.values()))


# class L0BEncodingSchema(BaseModel):
#     """Define the expected keys and values of the each variable in the l0b_encoding.yml file."""

#     dtype: str
#     zlib: bool
#     complevel: int
#     shuffle: bool
#     fletcher32: bool
#     contiguous: bool
#     _FillValue: Optional[Union[int, float]]
#     chunksizes: Union[int, List[int]]


# # Test the format and content of the l0b_encodings.yml files
# yaml_filepaths = list_files(CONFIG_FOLDER, "l0b_encodings.yml", recursive=True)


# @pytest.mark.parametrize("yaml_filepath", yaml_filepaths)
# def test_l0b_encodings_format(yaml_filepath: str) -> None:
#     """Test the l0b_encodings.yml file format.

#     Parameters
#     ----------
#     yaml_filepath : str
#         Path of the l0b_encodings.yml file to test.
#     """
#     data = read_yaml_file(yaml_filepath)
#     assert is_dict(data)
#     assert is_string_list(list(data.keys()))
#     # Check the second level of the dictionary match the schema
#     for value in data.values():
#         assert validate_schema_pytest(value, L0BEncodingSchema)
