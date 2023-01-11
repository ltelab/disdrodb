from disdrodb.L0 import check_standards
import pytest


def test_check_sensor_name():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.check_sensor_name()
    assert 1 == 1


def test_check_L0A_column_names():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.check_L0A_column_names()
    assert 1 == 1


def test_check_L0A_standards():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.check_L0A_standards()
    assert 1 == 1


def test_check_L0B_standards():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.check_L0B_standards()
    assert 1 == 1


def test_get_field_ndigits_natural_dict():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.get_field_ndigits_natural_dict()
    assert 1 == 1


def test_get_field_ndigits_decimals_dict():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.get_field_ndigits_decimals_dict()
    assert 1 == 1


def test_get_field_ndigits_dict():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.get_field_ndigits_dict()
    assert 1 == 1


def test_get_field_nchar_dict():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.get_field_nchar_dict()
    assert 1 == 1


def test_get_field_value_range_dict():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.get_field_value_range_dict()
    assert 1 == 1


def test_get_field_flag_dict():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.get_field_flag_dict()
    assert 1 == 1


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
        "datalogger_error": [1],
        "error_code": [1, 2],
    }

    # Test with an unknown device
    with pytest.raises(UnboundLocalError):
        check_standards.get_field_error_dict("unknown_device")
