#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:07:56 2022

@author: ghiggi
"""
import numpy as np
from disdrodb.L0.standards import (
    get_diameter_bin_center,
    get_diameter_bin_lower,
    get_diameter_bin_upper,
    get_diameter_bin_width,
    get_velocity_bin_center,
    get_velocity_bin_lower,
    get_velocity_bin_upper,
    get_velocity_bin_width,
    get_description_dict,
    get_units_dict,
    get_long_name_dict,
    get_data_format_dict,
    get_L0B_encodings_dict,
)

# ------------------------------------------------------------
# TODO:
# - function that check variables in L0A_dtypes match L0B_encodings.yml keys
# - check start diameter with OTT_Parsivel and OTT_Parsivel2
# ------------------------------------------------------------


def check_bin_consistency(sensor_name: str) -> None:
    """Check bin consistency from config file.

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

    np.testing.assert_allclose(
        diameter_bin_upper - diameter_bin_lower, diameter_bin_width
    )
    np.testing.assert_allclose(
        diameter_bin_lower + diameter_bin_width / 2, diameter_bin_center
    )
    np.testing.assert_allclose(
        diameter_bin_upper - diameter_bin_width / 2, diameter_bin_center
    )

    velocity_bin_lower = get_velocity_bin_lower(sensor_name)
    velocity_bin_upper = get_velocity_bin_upper(sensor_name)
    velocity_bin_center = get_velocity_bin_center(sensor_name)
    velocity_bin_width = get_velocity_bin_width(sensor_name)

    velocity_bin_lower = np.array(velocity_bin_lower)
    velocity_bin_upper = np.array(velocity_bin_upper)
    velocity_bin_center = np.array(velocity_bin_center)
    velocity_bin_width = np.array(velocity_bin_width)

    np.testing.assert_allclose(
        velocity_bin_upper - velocity_bin_lower, velocity_bin_width
    )
    np.testing.assert_allclose(
        velocity_bin_lower + velocity_bin_width / 2, velocity_bin_center
    )
    np.testing.assert_allclose(
        velocity_bin_upper - velocity_bin_width / 2, velocity_bin_center
    )


def check_variable_keys_consistency(sensor_name: str) -> None:
    """Check attributes consistency from config file.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    """
    description_dict = get_description_dict(sensor_name)
    units_dict = get_units_dict(sensor_name)
    long_name_dict = get_long_name_dict(sensor_name)
    data_format_dict = get_data_format_dict(sensor_name)
    encoding_dict = get_L0B_encodings_dict(sensor_name)
    encoding_vars = set(encoding_dict.keys())
    data_format_vars = set(data_format_dict.keys())
    long_name_vars = set(long_name_dict.keys())
    units_vars = set(units_dict.keys())
    description_vars = set(description_dict.keys())

    encoding_vars.difference(data_format_vars)
    encoding_vars.difference(units_vars)
    encoding_vars.difference(description_vars)
    encoding_vars.difference(long_name_vars)

    data_format_vars.difference(encoding_vars)
    units_vars.difference(encoding_vars)
    description_vars.difference(encoding_vars)
    long_name_vars.difference(encoding_vars)
