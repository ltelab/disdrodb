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


import pytest

from disdrodb.api.configs import available_sensor_names
from disdrodb.l0.check_configs import (
    _check_bin_consistency,
    _check_cf_attributes,
    _check_raw_array,
    _check_raw_data_format,
    _check_variable_consistency,
    _check_yaml_files_exists,
    check_all_sensors_configs,
    check_l0b_encoding,
)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test__check_yaml_files_exists(sensor_name):
    _check_yaml_files_exists(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test__check_variable_consistency(sensor_name):
    _check_variable_consistency(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_l0b_encoding(sensor_name):
    check_l0b_encoding(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_l0a_encoding(sensor_name):
    check_l0b_encoding(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test__check_raw_data_format(sensor_name):
    _check_raw_data_format(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test__check_cf_attributes(sensor_name):
    _check_cf_attributes(sensor_name)


# Test that all instruments config files are valid regarding the bin consistency
@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test__check_bin_consistency(sensor_name):
    # Test that no error is raised
    assert _check_bin_consistency(sensor_name) is None


# Test that all instruments config files are valid regarding the bin consistency
@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test__check_raw_array(sensor_name):
    # Test that no error is raised
    assert _check_raw_array(sensor_name) is None


def test_check_all_sensors_configs():
    # Test that no error is raised
    assert check_all_sensors_configs() is None
