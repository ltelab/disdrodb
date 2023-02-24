# The yaml files validity is tested in the test_config_files.py file
import os
import yaml
import pytest
import pandas as pd
from disdrodb.l0 import standards

# Set paths
ROOT_DISDRODB_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
)
CONFIG_FOLDER = os.path.join(ROOT_DISDRODB_FOLDER, "L0", "configs")


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_natural_dict(sensor_name):
    function_return = standards.get_field_ndigits_natural_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_decimals_dict(sensor_name):
    function_return = standards.get_field_ndigits_decimals_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_dict(sensor_name):
    function_return = standards.get_field_ndigits_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_nchar_dict(sensor_name):
    function_return = standards.get_field_nchar_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_data_range_dict(sensor_name):
    function_return = standards.get_data_range_dict(sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_nan_flags_dict(sensor_name):
    function_return = standards.get_nan_flags_dict(sensor_name)
    assert isinstance(function_return, dict)


"""

def test_read_config_yml():
    function_return = standards.read_config_yml()
    assert function_return ==


def test_get_configs_dir():
    function_return = standards.get_configs_dir()
    assert function_return ==

def test_available_sensor_name():
    function_return = standards.available_sensor_name()
    assert function_return ==

def test_get_variables_dict():
    function_return = standards.get_variables_dict()
    assert function_return ==

def test_get_sensor_variables():
    function_return = standards.get_sensor_variables()
    assert function_return ==

def test_get_data_format_dict():
    function_return = standards.get_data_format_dict()
    assert function_return ==

def test_get_description_dict():
    function_return = standards.get_description_dict()
    assert function_return ==

def test_get_long_name_dict():
    function_return = standards.get_long_name_dict()
    assert function_return ==

def test_get_units_dict():
    function_return = standards.get_units_dict()
    assert function_return ==

def test_get_diameter_bins_dict():
    function_return = standards.get_diameter_bins_dict()
    assert function_return ==

def test_get_velocity_bins_dict():
    function_return = standards.get_velocity_bins_dict()
    assert function_return ==

def test_get_l0a_dtype():
    function_return = standards.get_l0a_dtype()
    assert function_return ==

def test_get_L0A_encodings_dict():
    function_return = standards.get_L0A_encodings_dict()
    assert function_return ==

def test_get_L0B_encodings_dict():
    function_return = standards.get_L0B_encodings_dict()
    assert function_return ==

def test_get_time_encoding():
    function_return = standards.get_time_encoding()
    assert function_return ==

def test_set_DISDRODB_L0_attrs():
    function_return = standards.set_DISDRODB_L0_attrs()
    assert function_return ==

def test_get_diameter_bin_center():
    function_return = standards.get_diameter_bin_center()
    assert function_return ==

def test_get_diameter_bin_lower():
    function_return = standards.get_diameter_bin_lower()
    assert function_return ==

def test_get_diameter_bin_upper():
    function_return = standards.get_diameter_bin_upper()
    assert function_return ==

def test_get_diameter_bin_width():
    function_return = standards.get_diameter_bin_width()
    assert function_return ==

def test_get_velocity_bin_center():
    function_return = standards.get_velocity_bin_center()
    assert function_return ==

def test_get_velocity_bin_lower():
    function_return = standards.get_velocity_bin_lower()
    assert function_return ==

def test_get_velocity_bin_upper():
    function_return = standards.get_velocity_bin_upper()
    assert function_return ==

def test_get_velocity_bin_width():
    function_return = standards.get_velocity_bin_width()
    assert function_return ==

def test_get_raw_array_nvalues():
    function_return = standards.get_raw_array_nvalues()
    assert function_return ==

def test_get_raw_array_dims_order():
    function_return = standards.get_raw_array_dims_order()
    assert function_return ==

"""
