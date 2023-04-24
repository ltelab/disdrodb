import os
import pytest


from disdrodb.l0.check_configs import (
    check_bin_consistency,
    check_cf_attributes,
    check_raw_data_format,
    check_l0b_encoding,
    check_yaml_files_exists,
    check_variable_consistency,
)
from disdrodb.l0.standards import available_sensor_name

PATH_TEST_FOLDERS_FILES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytest_files")


@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_yaml_files_exists(sensor_name):
    check_yaml_files_exists(sensor_name)


@pytest.mark.xfail
@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_variable_consistency(sensor_name):
    check_variable_consistency(sensor_name)
    assert True


@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_l0b_encoding(sensor_name):
    check_l0b_encoding(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_l0a_encoding(sensor_name):
    check_l0b_encoding(sensor_name)


@pytest.mark.xfail
@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_raw_data_format(sensor_name):
    check_raw_data_format(sensor_name)


@pytest.mark.xfail
@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_cf_attributes(sensor_name):
    check_cf_attributes(sensor_name)


@pytest.mark.xfail
# Test that all instruments config files are valid regarding the bin consistency
@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_bin_consistency(sensor_name):
    # Test that no error is raised
    assert check_bin_consistency(sensor_name) is None
