import os
import pytest
import pandas as pd
from disdrodb.L0 import check_standards

# Set paths
ROOT_DISDRODB_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
)
CONFIG_FOLDER = os.path.join(ROOT_DISDRODB_FOLDER, "L0", "configs")


def test_check_raw_fields_available():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards._check_raw_fields_available()
    assert 1 == 1


def test_check_sensor_name():
    sensor_name = "wrong_sensor_name"
    # Test with an unknown device
    with pytest.raises(ValueError):
        check_standards.check_sensor_name(sensor_name)


def test_check_L0A_column_names():
    # not tested yet because relies on config files that can be modified

    # fake panda dataframe
    data = {"wrong_column_name": ["John", "Jane", "Bob", "Sara"]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        check_standards.check_L0A_column_names(df, sensor_name="OTT_Parsivel")


def test_check_L0A_standards():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.check_L0A_standards()
    assert 1 == 1


def test_check_L0B_standards():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.check_L0B_standards()
    assert 1 == 1


@pytest.mark.parametrize("list_sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_natural_dict(list_sensor_name):
    function_return = check_standards.get_field_ndigits_natural_dict(list_sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("list_sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_decimals_dict(list_sensor_name):
    function_return = check_standards.get_field_ndigits_decimals_dict(list_sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("list_sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_ndigits_dict(list_sensor_name):
    function_return = check_standards.get_field_ndigits_dict(list_sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("list_sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_nchar_dict(list_sensor_name):
    function_return = check_standards.get_field_nchar_dict(list_sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("list_sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_value_range_dict(list_sensor_name):
    function_return = check_standards.get_field_value_range_dict(list_sensor_name)
    assert isinstance(function_return, dict)


@pytest.mark.parametrize("list_sensor_name", os.listdir(CONFIG_FOLDER))
def test_get_field_flag_dict(list_sensor_name):
    function_return = check_standards.get_field_flag_dict(list_sensor_name)
    assert isinstance(function_return, dict)


def test_get_field_value_options_dict():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.get_field_value_options_dict()
    assert 1 == 1


def test_get_field_error_dict():
    # not tested yet because relies on config files that can be modified
    # Test with a known device
    known_device_error_dict = check_standards.get_field_error_dict("OTT_Parsivel")
    assert known_device_error_dict == {
        "sensor_status": [1, 2, 3],
        # "datalogger_error": [1],
        "error_code": [1, 2],
    }

    # Test with an unknown device
    with pytest.raises(UnboundLocalError):
        check_standards.get_field_error_dict("unknown_device")
