#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os

from typing import List, Union, Optional
from pydantic import BaseModel, ValidationError, validator

from disdrodb.l0.standards import available_sensor_name


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


CONFIG_FILES_LIST = [
    "bins_diameter.yml",
    "bins_velocity.yml",
    "l0a_encodings.yml",
    "l0b_encodings.yml",
    "raw_data_format.yml",
    "variables.yml",
    "variable_description.yml",
    "variable_long_name.yml",
    "variable_units.yml",
]


def check_yaml_files_exists(sensor_name: str) -> None:
    """Check if all config YAML files exist.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    config_dir = get_configs_dir(sensor_name)

    list_of_file_names = [os.path.split(i)[-1] for i in glob.glob(f"{config_dir}/*.yml")]

    missing_elements = set(CONFIG_FILES_LIST).difference(set(list_of_file_names))

    if missing_elements:
        missing_elements_text = ",".join(missing_elements)
        raise FileNotFoundError(f"Missing YAML files {missing_elements_text} in {config_dir} for sensor {sensor_name}.")


def check_variable_consistency(sensor_name: str) -> None:
    """
    Check variable consistency across config files.

    The variables specified within l0b_encoding.yml must be defined also in the other config files.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        If the keys are not consistent.
    """
    list_variables = read_config_yml(sensor_name, "l0b_encodings.yml").keys()

    list_to_check = [
        "l0a_encodings.yml",
        "raw_data_format.yml",
        "variable_description.yml",
        "variable_long_name.yml",
        "variable_units.yml",
    ]

    for file_name in list_to_check:
        keys_to_check = read_config_yml(sensor_name, file_name).keys()
        missing_elements = set(list_variables).difference(set(keys_to_check))
        missing_elements_text = ",".join(missing_elements)
        extra_elements = set(keys_to_check).difference(set(list_variables))
        extra_elements_text = ",".join(extra_elements)
        if missing_elements:
            raise ValueError(
                f"Variable keys {missing_elements_text} (missing) and {extra_elements_text} (extra) are not consistent"
                f" in {file_name} for sensor {sensor_name}."
            )


class SchemaValidationException(Exception):
    """Exception raised when schema validation fails"""

    pass


def schema_error(object_to_validate: Union[str, list], schema: BaseModel, message) -> bool:
    """Function that validate the schema of a given object with a given schema.

    Parameters
    ----------
    object_to_validate : Union[str,list]
        Object to validate
    schema : BaseModel
        Base model

    """

    try:
        schema(**object_to_validate)

    except ValidationError as e:
        for i in e.errors():
            raise SchemaValidationException(f"Schema validation failed. {message} {e}")


class NetcdfEncodingSchema(BaseModel):
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
            raise ValueError("'chunksizes' must be defined if 'contiguous' is False")
        return v

    # if contiguous = True, then zlib must be set to False
    @validator("zlib")
    def check_zlib(cls, v, values):
        if values.get("contiguous") and v:
            raise ValueError("'zlib' must be set to False if 'contiguous' is True")
        return v

    # if contiguous = True, then fletcher32 must be set to False
    @validator("fletcher32")
    def check_fletcher32(cls, v, values):
        if values.get("contiguous") and v:
            raise ValueError("'fletcher32' must be set to False if 'contiguous' is True")
        return v


def check_l0b_encoding(sensor_name: str) -> None:
    """Check l0b_encodings.yml file based on the schema defined in the class NetcdfEncodingSchema.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """

    data = read_config_yml(sensor_name, "l0b_encodings.yml")

    # check that the second level of the dictionary match the schema
    for key, value in data.items():
        schema_error(
            object_to_validate=value,
            schema=NetcdfEncodingSchema,
            message=f"Sensore name : {sensor_name}. Key : {key}.",
        )


def check_l0a_encoding(sensor_name: str) -> None:
    """Check l0a_encodings.yml file.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        Error raised if the value of a key is not in the list of accepted values.
    """
    data = read_config_yml(sensor_name, "l0a_encodings.yml")

    numeric_field = ["float32", "uint32", "uint16", "uint8", "float64", "uint64", "int8", "int16", "int32", "int64"]

    text_field = ["str"]

    for key, value in data.items():
        if value not in text_field + numeric_field:
            raise ValueError(f"Wrong value for {key} in l0a_encodings.yml for sensor {sensor_name}.")


class RawDataFormatSchema(BaseModel):
    n_digits: Optional[int]
    n_characters: Optional[int]
    n_decimals: Optional[int]
    n_naturals: Optional[int]
    data_range: Optional[List[float]]
    nan_flags: Optional[str]
    valid_values: Optional[List[float]]
    dimension_order: Optional[List[str]]
    n_values: Optional[int]

    @validator("data_range", pre=True)
    def check_list_length(cls, value):
        if value:
            if len(value) != 2:
                raise ValueError(f"data_range must have exactly 2 elements, {len(value)} element have been provided.")
            return value


def check_raw_data_format(sensor_name: str) -> None:
    """check raw_data_format.yml file based on the schema defined in the class RawDataFormatSchema.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    data = read_config_yml(sensor_name, "raw_data_format.yml")

    # check that the second level of the dictionary match the schema
    for key, value in data.items():
        schema_error(
            object_to_validate=value,
            schema=RawDataFormatSchema,
            message=f"Sensore name : {sensor_name}. Key : {key}.",
        )


def check_cf_attributes(sensor_name: str) -> None:
    """Check that variable_description, variable_long_name, variable_units dict values are strings.

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

    diameter_bin_lower = np.array(get_diameter_bin_lower(sensor_name))
    diameter_bin_upper = np.array(get_diameter_bin_upper(sensor_name))
    diameter_bin_center = np.array(get_diameter_bin_center(sensor_name))
    diameter_bin_width = np.array(get_diameter_bin_width(sensor_name))

    expected_diameter_width = diameter_bin_upper - diameter_bin_lower
    np.testing.assert_allclose(expected_diameter_width[1:-1], diameter_bin_width[1:-1], atol=1e-3, rtol=1e-4)

    expected_diameter_center = diameter_bin_lower + diameter_bin_width / 2
    np.testing.assert_allclose(expected_diameter_center[1:-1], diameter_bin_center[1:-1], atol=1e-3, rtol=1e-4)

    expected_diameter_center = diameter_bin_upper - diameter_bin_width / 2
    np.testing.assert_allclose(expected_diameter_center[1:-1], diameter_bin_center[1:-1], atol=1e-3, rtol=1e-4)

    velocity_bin_lower = np.array(get_velocity_bin_lower(sensor_name))
    velocity_bin_upper = np.array(get_velocity_bin_upper(sensor_name))
    velocity_bin_center = np.array(get_velocity_bin_center(sensor_name))
    velocity_bin_width = np.array(get_velocity_bin_width(sensor_name))

    if all(arr.size > 1 for arr in [velocity_bin_center, velocity_bin_lower, velocity_bin_upper, velocity_bin_width]):
        np.testing.assert_allclose(velocity_bin_upper - velocity_bin_lower, velocity_bin_width, atol=1e-3, rtol=1e-4)
        np.testing.assert_allclose(
            velocity_bin_lower + velocity_bin_width / 2, velocity_bin_center, atol=1e-3, rtol=1e-4
        )
        np.testing.assert_allclose(
            velocity_bin_upper - velocity_bin_width / 2, velocity_bin_center, atol=1e-3, rtol=1e-4
        )


def get_bins_measurement(sensor_name: str, file_name: str) -> list:
    """get bins measurement from config file.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    file_name : str
        File name (bins_velocity.yml or bins_diameter.yml)

    Returns
    -------
    list
        List of chunksizes (center, bounds, width)
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
        Error if the chunksizes are not consistent.
    """
    raw_data_format = read_config_yml(sensor_name, "raw_data_format.yml")

    # Get keys in raw_data_format where the value is "dimension_order"
    dict_keys_with_dimension_order = {
        key: value.get("dimension_order") for key, value in raw_data_format.items() if "dimension_order" in value.keys()
    }

    l0b_encodings = read_config_yml(sensor_name, "l0b_encodings.yml")

    for key, list_velocity_or_diameter in dict_keys_with_dimension_order.items():
        expected_lenght = len(list_velocity_or_diameter) + 1
        current_length = len(l0b_encodings.get(key).get("chunksizes"))
        if expected_lenght != current_length:
            raise ValueError(f"Wrong chunksizes for {key} in l0b_encodings.yml for sensor {sensor_name}.")

    # Get chunksizes in l0b_encoding.yml and check that if len > 1, has dimension_order key in raw_data_format
    list_attributes_L0B_encodings = [
        i
        for i in l0b_encodings.keys()
        if isinstance(l0b_encodings.get(i).get("chunksizes"), list) and len(l0b_encodings.get(i).get("chunksizes")) > 1
    ]
    list_attributes_from_raw_data_format = [
        i for i in raw_data_format.keys() if raw_data_format.get(i).get("dimension_order") is not None
    ]

    if not sorted(list_attributes_L0B_encodings) == sorted(list_attributes_from_raw_data_format):
        raise ValueError(f"Chunksizes in l0b_encodings and raw_data_format for sensor {sensor_name} does not match.")


def check_sensor_configs(sensor_name: str) -> None:
    """check sensor configs.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    check_yaml_files_exists(sensor_name)
    check_variable_consistency(sensor_name)
    check_l0b_encoding(sensor_name=sensor_name)
    check_l0a_encoding(sensor_name=sensor_name)
    check_raw_data_format(sensor_name=sensor_name)
    check_cf_attributes(sensor_name=sensor_name)
    check_bin_consistency(sensor_name=sensor_name)
    check_raw_array(sensor_name=sensor_name)


def check_all_sensors_configs() -> None:
    """Check all sensors configs."""
    for sensor_name in available_sensor_name():
        check_sensor_configs(sensor_name=sensor_name)
