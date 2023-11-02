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
"""Test DISDRODB L0 standards routines."""

import os

import pytest

from disdrodb import __root_path__
from disdrodb.l0.standards import (
    get_data_range_dict,
    get_field_nchar_dict,
    get_field_ndigits_decimals_dict,
    get_field_ndigits_dict,
    get_field_ndigits_natural_dict,
    get_nan_flags_dict,
    get_time_encoding,
    get_valid_coordinates_names,
    get_valid_dimension_names,
    get_valid_names,
    get_valid_variable_names,
    get_variables_dimension,
)

CONFIG_FOLDER = os.path.join(__root_path__, "disdrodb", "l0", "configs")


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_natural_dict(sensor_name):
    function_return = get_field_ndigits_natural_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_decimals_dict(sensor_name):
    function_return = get_field_ndigits_decimals_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_dict(sensor_name):
    function_return = get_field_ndigits_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_nchar_dict(sensor_name):
    function_return = get_field_nchar_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_data_range_dict(sensor_name):
    function_return = get_data_range_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_nan_flags_dict(sensor_name):
    function_return = get_nan_flags_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_variables_dimension(sensor_name):
    assert isinstance(get_variables_dimension(sensor_name), dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_valid_variable_names(sensor_name):
    assert isinstance(get_valid_variable_names(sensor_name), list)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_valid_dimension_names(sensor_name):
    assert isinstance(get_valid_dimension_names(sensor_name), list)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_valid_coordinates_names(sensor_name):
    assert isinstance(get_valid_coordinates_names(sensor_name), list)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_valid_names(sensor_name):
    assert isinstance(get_valid_names(sensor_name), list)


def test_get_time_encoding():
    assert isinstance(get_time_encoding(), dict)
