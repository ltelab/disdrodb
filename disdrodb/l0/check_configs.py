#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os

from typing import List, Union, Optional
from pydantic import BaseModel, ValidationError, validator


import numpy as np
from disdrodb.l0.standards import (
    get_diameter_bin_center,
    get_diameter_bin_lower,
    get_diameter_bin_upper,
    get_diameter_bin_width,
    get_velocity_bin_center,
    get_velocity_bin_lower,
    get_velocity_bin_upper,
    get_velocity_bin_width,
    get_configs_dir,
    read_config_yml,
)

# ------------------------------------------------------------
# TODO:
# - function that check variables in L0A_dtypes match L0B_encodings.yml keys
# - check start diameter with OTT_Parsivel and OTT_Parsivel2
# ------------------------------------------------------------

CONFIG_FILES_LIST = [
    "bins_diameter.yml",
    "bins_velocity.yml",
    "L0A_encodings.yml",
    "L0B_encodings.yml",
    "raw_data_format.yml",
    "variables.yml",
    "variable_description.yml",
    "variable_long_name.yml",
    "variable_units.yml",
]


def check_yaml_files_exists(sensor_name: str) -> None:
    """Check if all yaml files exist.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    config_dir = get_configs_dir(sensor_name)

    list_of_file_names = [os.path.split(i)[-1] for i in glob.glob(f"{config_dir}/*.yml")]

    if not list_of_file_names == CONFIG_FILES_LIST:
        raise FileNotFoundError(f"Missing yaml files in {config_dir} for sensor {sensor_name}.")


def check_variable_consistency(sensor_name: str) -> None:
    """Check variable consistency from config file.
    The values inside variables.yml must be consistent with the other files.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        If the keys are not consistent.
    """
    list_variables = read_config_yml(sensor_name, "variables.yml").values()

    list_to_check = [
        "L0A_encodings.yml",
        "L0B_encodings.yml",
        "raw_data_format.yml",
        "variable_description.yml",
        "variable_long_name.yml",
        "variable_units.yml",
    ]

    for file_name in list_to_check:
        keys_to_check = read_config_yml(sensor_name, file_name).keys()
        if not sorted(list(list_variables)) == sorted(list(keys_to_check)):
            raise ValueError(f"Variable keys are not consistent in {file_name} for sensor {sensor_name}.")


class SchemaValidationException(Exception):
    """Exception raised when schema validation fails"""

    pass


def schema_error(schema_to_validate: Union[str, list], schema: BaseModel, message) -> bool:
    """function that validate the schema of a given object with a given schema.

    Parameters
    ----------
    schema_to_validate : Union[str,list]
        Object to validate
    schema : BaseModel
        Base model

    """

    try:
        schema(**schema_to_validate)

    except ValidationError as e:
        e.errors()[0]["loc"][0]

        for i in e.errors():
            raise SchemaValidationException(f"Schema validation failed. {message} {e}")


class L0B_encodings_2n_level(BaseModel):
    contiguous: bool
    dtype: str
    zlib: bool
    complevel: int
    shuffle: bool
    fletcher32: bool
    _FillValue: Optional[Union[int, float]]
    chunksizes: Optional[Union[int, List[int]]]

    # if contiguous=False, chunksizes specified, otherwise should be not !
    @validator("chunksizes")
    def check_chunksizes(cls, v, values):
        if not values.get("contiguous") and not v:
            raise ValueError("chunksizes must be defined if contiguous is False")
        return v

    # if contiguous = True, then zlib must be set to False
    @validator("zlib")
    def check_zlib(cls, v, values):
        if values.get("contiguous") and v:
            raise ValueError("zlib must be set to False if contiguous is True")
        return v

    # if contiguous = True, then fletcher32 must be set to False
    @validator("fletcher32")
    def check_fletcher32(cls, v, values):
        if values.get("contiguous") and v:
            raise ValueError("fletcher32 must be set to False if contiguous is True")
        return v


def check_l0b_encoding(sensor_name: str) -> None:
    """Check L0B_encodings.yml file based on the schema defined in the class L0B_encodings_2n_level.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """

    data = read_config_yml(sensor_name, "L0B_encodings.yml")

    # check that the second level of the dictionary match the schema
    for key, value in data.items():
        schema_error(
            schema_to_validate=value,
            schema=L0B_encodings_2n_level,
            message=f"Sensore name : {sensor_name}. Key : {key}.",
        )


def check_l0a_encoding(sensor_name: str) -> None:
    """Check L0A_encodings.yml file.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        Error raised if the value of a key is not in the list of accepted values.
    """
    data = read_config_yml(sensor_name, "L0A_encodings.yml")

    numeric_field = ["float32", "uint32", "uint16", "uint8"]

    text_field = ["str"]

    for key, value in data.items():
        if value not in text_field + numeric_field:
            raise ValueError(f"Wrong value for {key} in L0A_encodings.yml for sensor {sensor_name}.")


class raw_data_format_2n_level(BaseModel):
    n_digits: int
    n_characters: int
    n_decimals: int
    n_naturals: int
    data_range: Optional[List[float]]
    nan_flags: Optional[str]
    valid_values: Optional[List[float]]
    dimension_order: Optional[List[str]]
    n_values: Optional[int]

    @validator("data_range", pre=True)
    def check_list_length(cls, value):
        if value:
            if len(value) != 2:
                raise ValueError("my_list must have exactly 2 elements")
            return value


def check_raw_data_format(sensor_name: str) -> None:
    """check raw_data_format.yml file based on the schema defined in the class raw_data_format_2n_level.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    data = read_config_yml(sensor_name, "raw_data_format.yml")

    # check that the second level of the dictionary match the schema
    for key, value in data.items():
        schema_error(
            schema_to_validate=value,
            schema=raw_data_format_2n_level,
            message=f"Sensore name : {sensor_name}. Key : {key}.",
        )


def check_cf_attributes(sensor_name: str) -> None:
    """check that variable_description, variable_long_name, variable_units dict values are strings.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    list_of_files = ["variable_description.yml", "variable_long_name.yml", "variable_units.yml"]
    for file in list_of_files:
        data = read_config_yml(sensor_name, file)
        for key, value in data.items():
            if not isinstance(value, str):
                raise ValueError(f"Wrong value for {key} in {file} for sensor {sensor_name}.")


def check_bin_consistency(sensor_name: str) -> None:
    """Check bin consistency from config file.

    Do not check the first and last bin !

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """

    diameter_bin_lower = get_diameter_bin_lower(sensor_name)
    diameter_bin_upper = get_diameter_bin_upper(sensor_name)
    diameter_bin_center = get_diameter_bin_center(sensor_name)
    diameter_bin_width = get_diameter_bin_width(sensor_name)
    diameter_bin_lower = np.array(diameter_bin_lower)
    diameter_bin_upper = np.array(diameter_bin_upper)
    diameter_bin_center = np.array(diameter_bin_center)
    diameter_bin_width = np.array(diameter_bin_width)

    expected_diameter_width = diameter_bin_upper - diameter_bin_lower
    np.testing.assert_allclose(expected_diameter_width[1:-1], diameter_bin_width[1:-1])

    expected_diameter_center = diameter_bin_lower + diameter_bin_width / 2
    np.testing.assert_allclose(expected_diameter_center[1:-1], diameter_bin_center[1:-1])

    expected_diameter_center = diameter_bin_upper - diameter_bin_width / 2
    np.testing.assert_allclose(expected_diameter_center[1:-1], diameter_bin_center[1:-1])

    velocity_bin_lower = get_velocity_bin_lower(sensor_name)
    velocity_bin_upper = get_velocity_bin_upper(sensor_name)
    velocity_bin_center = get_velocity_bin_center(sensor_name)
    velocity_bin_width = get_velocity_bin_width(sensor_name)

    velocity_bin_lower = np.array(velocity_bin_lower)
    velocity_bin_upper = np.array(velocity_bin_upper)
    velocity_bin_center = np.array(velocity_bin_center)
    velocity_bin_width = np.array(velocity_bin_width)

    np.testing.assert_allclose(velocity_bin_upper - velocity_bin_lower, velocity_bin_width)
    np.testing.assert_allclose(velocity_bin_lower + velocity_bin_width / 2, velocity_bin_center)
    np.testing.assert_allclose(velocity_bin_upper - velocity_bin_width / 2, velocity_bin_center)


def get_chunksizes(sensor_name: str, file_name: str) -> list:
    """get chunksizes from config file.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    file_name : str
        File name (bins_velocity.yml or bins_diameter.yml)

    Returns
    -------
    list
        list of chunksizes (center, bounds, width)
    """
    data = read_config_yml(sensor_name, file_name)
    center_len = len(data.get("center"))
    bound_len = len(data.get("bounds"))
    width_len = len(data.get("width"))

    return [center_len, bound_len, width_len]


def check_raw_array(sensor_name: str) -> None:
    """Check raw array consistency from config file.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        Error if the chuncksizes are not consistent.
    """
    raw_data_format = read_config_yml(sensor_name, "raw_data_format.yml")

    # Get keys in raw_data_format where the value is "dimension_order"
    dict_keys_with_dimension_order = {
        key: value.get("dimension_order") for key, value in raw_data_format.items() if "dimension_order" in value.keys()
    }

    # Get chunksizes
    L0B_encodings = read_config_yml(sensor_name, "L0B_encodings.yml")

    # Iterate over raw_data_format keys with "dimension_order" value
    for key, list_velocity_or_diameter in dict_keys_with_dimension_order.items():
        for i, velocity_or_diameter in enumerate(list_velocity_or_diameter):
            # get the definiation of the chunksizes
            chunksize_definition = L0B_encodings.get(key).get("chunksizes")[i + 1]

            # define config file name
            file_name = "bins_velocity.yml" if "velocity" in velocity_or_diameter else "bins_diameter.yml"

            # get the chunksizes from config file
            chunksize = get_chunksizes(sensor_name=sensor_name, file_name=file_name)

            # raise a exception if all chunksize are not equal to the definition
            if not all(x == chunksize_definition for x in chunksize):
                raise ValueError(f"Wrong value for {key} in {file_name} for sensor {sensor_name}.")

    # Get chunksizes in chunksizes in l0b_encoding and check that if len > 1, has dimension_order key in raw_data_format
    list_attributes_L0B_encodings = [
        i
        for i in L0B_encodings.keys()
        if isinstance(L0B_encodings.get(i).get("chunksizes"), list) and len(L0B_encodings.get(i).get("chunksizes")) > 1
    ]
    list_attribites_from_raw_data_format = [
        i for i in raw_data_format.keys() if raw_data_format.get(i).get("dimension_order") is not None
    ]

    if not sorted(list_attributes_L0B_encodings) == sorted(list_attribites_from_raw_data_format):
        raise ValueError(f"Chunksizes in L0B_encodings and raw_data_format for sensor {sensor_name} does not match.")
