# The yaml files validity is tested in the test_config_files.py file
import os
import pytest
from disdrodb.l0.standards import (
    get_time_encoding,
    get_field_ndigits_natural_dict,
    get_field_ndigits_decimals_dict,
    get_field_ndigits_dict,
    get_field_nchar_dict,
    get_data_range_dict,
    get_nan_flags_dict,
    get_variables_dimension,
    get_valid_variable_names,
    get_valid_dimension_names,
    get_valid_names,
    get_valid_coordinates_names,
)

# Set paths
ROOT_DISDRODB_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
)
CONFIG_FOLDER = os.path.join(ROOT_DISDRODB_FOLDER, "l0", "configs")


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


def test_get_time_encoding():
    assert isinstance(get_time_encoding(), dict)


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
