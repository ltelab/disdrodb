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
"""Check configuration files."""

import os
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from disdrodb.api.configs import available_sensor_names, get_sensor_configs_dir, read_config_file
from disdrodb.l0.standards import (
    get_diameter_bin_center,
    get_diameter_bin_lower,
    get_diameter_bin_upper,
    get_diameter_bin_width,
    get_velocity_bin_center,
    get_velocity_bin_lower,
    get_velocity_bin_upper,
    get_velocity_bin_width,
)
from disdrodb.utils.directories import list_files

CONFIG_FILES_LIST = [
    "bins_diameter.yml",
    "bins_velocity.yml",
    "raw_data_format.yml",
    "l0a_encodings.yml",
    "l0b_encodings.yml",
    "l0b_cf_attrs.yml",
]


def _check_yaml_files_exists(sensor_name: str) -> None:
    """Check if all L0 config YAML files exist.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    config_dir = get_sensor_configs_dir(sensor_name, product="L0A")
    filepaths = list_files(config_dir, glob_pattern="*.yml", recursive=False)
    filenames = [os.path.split(i)[-1] for i in filepaths]
    missing_keys = set(CONFIG_FILES_LIST).difference(set(filenames))
    if missing_keys:
        missing_keys_text = ",".join(missing_keys)
        raise FileNotFoundError(f"Missing YAML files {missing_keys_text} in {config_dir} for sensor {sensor_name}.")


def _check_variable_consistency(sensor_name: str) -> None:
    """
    Check variable consistency across config files.

    The variables specified within l0b_encoding.yml must be defined also in the other config files.
    The raw_data_format.yml can contain some extra variables !

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        If the keys are not consistent.
    """
    list_variables = read_config_file(sensor_name, product="L0A", filename="l0b_cf_attrs.yml").keys()
    filenames = [
        "raw_data_format.yml",
        "l0a_encodings.yml",
        "l0b_encodings.yml",
    ]
    for filename in filenames:
        keys_to_check = read_config_file(sensor_name, product="L0A", filename=filename).keys()
        missing_keys = set(list_variables).difference(set(keys_to_check))
        extra_keys = set(keys_to_check).difference(set(list_variables))
        if missing_keys:
            msg = f"The {sensor_name} {filename} file does not have the following keys {missing_keys}"
            if extra_keys:
                if filename == "raw_data_format.yml":
                    msg = msg + f"FYI the file has the following extra keys {extra_keys}."
                else:
                    msg = msg + f"and it has the following extra keys {extra_keys}."
            raise ValueError(msg)
        if extra_keys and filename != "raw_data_format.yml":
            msg = f"The {sensor_name} {filename} has the following extra keys {extra_keys}."
            raise ValueError(msg)


class SchemaValidationException(Exception):
    """Exception raised when schema validation fails."""


def _schema_error(object_to_validate: Union[str, list], schema: BaseModel, message) -> bool:
    """Function that validate the schema of a given object with a given schema.

    Parameters
    ----------
    object_to_validate : Union[str,list]
        Object to validate.
    schema : BaseModel
        Base model.

    """
    try:
        schema(**object_to_validate)
    except ValidationError as e:
        raise SchemaValidationException(f"Schema validation failed. {message} {e}")


class L0BEncodingSchema(BaseModel):
    """Pydantic model for DISDRODB L0B encodings."""

    contiguous: bool
    dtype: str
    zlib: bool
    complevel: int
    shuffle: bool
    fletcher32: bool
    _FillValue: Optional[Union[int, float]]
    chunksizes: Optional[Union[int, list[int]]]

    # if contiguous=False, chunksizes specified, otherwise should be not !
    @model_validator(mode="before")
    def check_chunksizes_and_zlib(cls, values):
        """Check the chunksizes validity."""
        contiguous = values.get("contiguous")
        chunksizes = values.get("chunksizes")
        if not contiguous and not chunksizes:
            raise ValueError("'chunksizes' must be defined if 'contiguous' is False")
        return values

    # if contiguous = True, then zlib must be set to False
    @model_validator(mode="before")
    def check_contiguous_and_zlib(cls, values):
        """Check the the compression value validity."""
        contiguous = values.get("contiguous")
        zlib = values.get("zlib")
        if contiguous and zlib:
            raise ValueError("'zlib' must be set to False if 'contiguous' is True")
        return values

    # if contiguous = True, then fletcher32 must be set to False
    @model_validator(mode="before")
    def check_contiguous_and_fletcher32(cls, values):
        """Check the fletcher value validity."""
        contiguous = values.get("contiguous")
        fletcher32 = values.get("fletcher32")
        if contiguous and fletcher32:
            raise ValueError("'fletcher32' must be set to False if 'contiguous' is True")
        return values


def check_l0b_encoding(sensor_name: str) -> None:
    """Check ``l0b_encodings.yml`` file based on the schema defined in the class ``L0BEncodingSchema``.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    data = read_config_file(sensor_name, product="L0A", filename="l0b_encodings.yml")

    # check that the second level of the dictionary match the schema
    for key, value in data.items():
        _schema_error(
            object_to_validate=value,
            schema=L0BEncodingSchema,
            message=f"Sensore name : {sensor_name}. Key : {key}.",
        )


def check_l0a_encoding(sensor_name: str) -> None:
    """Check ``l0a_encodings.yml`` file.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Raises
    ------
    ValueError
        Error raised if the value of a key is not in the list of accepted values.
    """
    data = read_config_file(sensor_name, product="L0A", filename="l0a_encodings.yml")
    numeric_field = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64"]
    text_field = ["str"]
    for key, value in data.items():
        if value not in text_field + numeric_field:
            raise ValueError(f"Wrong value for {key} in l0a_encodings.yml for sensor {sensor_name}.")
        if not isinstance(key, str):
            raise TypeError(f"Expecting a string for {key} in l0a_encodings.yml for sensor {sensor_name}.")


class RawDataFormatSchema(BaseModel):
    """Pydantic model for the DISDRODB RAW Data Format YAML files."""

    n_digits: Optional[int]
    n_characters: Optional[int]
    n_decimals: Optional[int]
    n_naturals: Optional[int]
    data_range: Optional[list[float]]
    nan_flags: Optional[Union[int, str]] = None
    valid_values: Optional[list[float]] = None
    dimension_order: Optional[list[str]] = None
    n_values: Optional[int] = None
    field_number: Optional[str] = None

    @field_validator("data_range")
    def check_list_length(cls, value):
        """Check the data_range validity."""
        if value:
            if len(value) != 2:
                raise ValueError(f"data_range must have exactly 2 keys, {len(value)} element have been provided.")
            return value
        return None


def _check_raw_data_format(sensor_name: str) -> None:
    """Check ``raw_data_format.yml`` file based on the schema defined in the class ``RawDataFormatSchema``.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    data = read_config_file(sensor_name, product="L0A", filename="raw_data_format.yml")

    # check that the second level of the dictionary match the schema
    for key, value in data.items():
        _schema_error(
            object_to_validate=value,
            schema=RawDataFormatSchema,
            message=f"Sensore name : {sensor_name}. Key : {key}.",
        )


def _check_cf_attributes(sensor_name: str) -> None:
    """Check that the ``l0b_cf_attrs.yml`` description, long_name and units values are strings.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    cf_dict = read_config_file(sensor_name, product="L0A", filename="l0b_cf_attrs.yml")
    for var, attrs_dict in cf_dict.items():
        for key, value in attrs_dict.items():
            if not isinstance(value, str):
                raise ValueError(f"Wrong value for {key} in {var} for sensor {sensor_name}.")


def _check_bin_consistency(sensor_name: str) -> None:
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
            velocity_bin_lower + velocity_bin_width / 2,
            velocity_bin_center,
            atol=1e-3,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            velocity_bin_upper - velocity_bin_width / 2,
            velocity_bin_center,
            atol=1e-3,
            rtol=1e-4,
        )


def _check_raw_array(sensor_name: str) -> None:
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
    raw_data_format = read_config_file(sensor_name, product="L0A", filename="raw_data_format.yml")

    # Get keys in raw_data_format where the value is "dimension_order"
    dict_keys_with_dimension_order = {
        key: value.get("dimension_order") for key, value in raw_data_format.items() if "dimension_order" in value
    }

    l0b_encodings = read_config_file(sensor_name, product="L0A", filename="l0b_encodings.yml")

    for key, list_velocity_or_diameter in dict_keys_with_dimension_order.items():
        expected_length = len(list_velocity_or_diameter) + 1
        current_length = len(l0b_encodings.get(key).get("chunksizes"))
        if expected_length != current_length:
            raise ValueError(f"Wrong chunksizes for {key} in l0b_encodings.yml for sensor {sensor_name}.")

    # Get chunksizes in l0b_encoding.yml and check that if len > 1, has dimension_order key in raw_data_format
    list_attributes_l0b_encodings = [
        i
        for i in l0b_encodings
        if isinstance(l0b_encodings.get(i).get("chunksizes"), list) and len(l0b_encodings.get(i).get("chunksizes")) > 1
    ]
    list_attributes_from_raw_data_format = [
        i for i in raw_data_format if raw_data_format.get(i).get("dimension_order") is not None
    ]

    if not sorted(list_attributes_l0b_encodings) == sorted(list_attributes_from_raw_data_format):
        raise ValueError(f"Chunksizes in l0b_encodings and raw_data_format for sensor {sensor_name} does not match.")


def check_sensor_configs(sensor_name: str) -> None:
    """Check validity of sensor configuration YAML files.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    _check_yaml_files_exists(sensor_name)
    _check_variable_consistency(sensor_name)
    check_l0b_encoding(sensor_name=sensor_name)
    check_l0a_encoding(sensor_name=sensor_name)
    _check_raw_data_format(sensor_name=sensor_name)
    _check_cf_attributes(sensor_name=sensor_name)
    _check_bin_consistency(sensor_name=sensor_name)
    _check_raw_array(sensor_name=sensor_name)


def check_all_sensors_configs() -> None:
    """Check all sensors configuration YAML files."""
    for sensor_name in available_sensor_names():
        check_sensor_configs(sensor_name=sensor_name)
