import pytest
import pandas as pd
from disdrodb.l0 import check_standards


def test_check_raw_fields_available():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards._check_raw_fields_available()
    assert 1 == 1


def test_check_sensor_name():
    sensor_name = "wrong_sensor_name"
    # Test with an unknown device
    with pytest.raises(ValueError):
        check_standards.check_sensor_name(sensor_name)


def test_check_l0a_column_names():
    # not tested yet because relies on config files that can be modified

    # fake panda dataframe
    data = {"wrong_column_name": ["John", "Jane", "Bob", "Sara"]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        check_standards.check_l0a_column_names(df, sensor_name="OTT_Parsivel")


def test_check_l0a_standards():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.check_l0a_standards()
    assert 1 == 1


def test_check_l0b_standards():
    # not tested yet because relies on config files that can be modified
    # function_return = check_standards.check_l0b_standards()
    assert 1 == 1
