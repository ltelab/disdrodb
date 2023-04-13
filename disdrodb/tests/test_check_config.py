import os
import pytest


from disdrodb.l0 import check_configs
from disdrodb.l0.standards import available_sensor_name

PATH_TEST_FOLDERS_FILES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytest_files")


# Test that all instruments config files are valid regarding the bin consistency
@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_bin_consistency(sensor_name):
    # Test that no error is raised
    assert check_configs.check_bin_consistency(sensor_name) is None


# Test that all instruments config files are valid regarding the keys consistency
@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_variable_keys_consistency(sensor_name):
    # Test that no error is raised
    assert check_configs.check_variable_keys_consistency(sensor_name) is None


# Test that overall tests are valid
@pytest.mark.parametrize("sensor_name", available_sensor_name())
def test_check_sensor_configs(sensor_name):
    # Test that no error is raised
    assert check_configs.check_sensor_configs(sensor_name) is None
