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
    _ensure_valid_netcdf_encoding_dict,
    get_data_range_dict,
    get_field_nchar_dict,
    get_field_ndigits_decimals_dict,
    get_field_ndigits_dict,
    get_field_ndigits_natural_dict,
    get_l0a_encodings_dict,
    get_n_velocity_bins,
    get_nan_flags_dict,
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


def test_get_n_velocity_bins():
    # Impact disdrometer
    sensor_name = "RD_80"
    assert get_n_velocity_bins(sensor_name) == 0

    # Optical disdrometer
    sensor_name = "OTT_Parsivel"
    assert get_n_velocity_bins(sensor_name) > 1


def test_ensure_valid_netcdf_encoding_dict():
    # Check raise error if contiguous=True but chunksize is specified
    encoding_dict = {}
    encoding_dict["var"] = {"contiguous": True, "chunksizes": [1, 2]}
    with pytest.raises(ValueError):
        _ensure_valid_netcdf_encoding_dict(encoding_dict)

    # Check correct encodings when contiguous array (only contiguous specified¨)
    encoding_dict = {}
    encoding_dict["var"] = {"contiguous": False}
    output_dict = _ensure_valid_netcdf_encoding_dict(encoding_dict)
    assert not output_dict["var"]["fletcher32"]
    assert not output_dict["var"]["zlib"]  # this might be relaxed
    assert output_dict["var"]["chunksizes"] == []

    # Check set contiguous=True encodings when contiguous=False but chunksizes is None
    encoding_dict = {}
    encoding_dict["var"] = {"contiguous": False, "chunksizes": None}
    output_dict = _ensure_valid_netcdf_encoding_dict(encoding_dict)
    assert output_dict["var"]["contiguous"]  # changed here !
    assert not output_dict["var"]["fletcher32"]
    assert not output_dict["var"]["zlib"]  # this might be relaxed
    assert output_dict["var"]["chunksizes"] == []

    # Check chunksize is list when contiguous=False
    encoding_dict = {}
    encoding_dict["var"] = {"contiguous": False, "chunksizes": 5}
    output_dict = _ensure_valid_netcdf_encoding_dict(encoding_dict)
    assert output_dict["var"]["chunksizes"] == [5]

    encoding_dict = {}
    encoding_dict["var"] = {"contiguous": False, "chunksizes": [1, 2]}
    output_dict = _ensure_valid_netcdf_encoding_dict(encoding_dict)
    assert output_dict["var"]["chunksizes"] == [1, 2]


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_l0a_encodings_dict(sensor_name):
    function_return = get_l0a_encodings_dict(sensor_name)
    assert isinstance(function_return, dict)
