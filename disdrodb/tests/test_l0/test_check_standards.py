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
"""Test DISDRODB L0 standards."""

import os

import numpy as np
import pandas as pd
import pytest

from disdrodb import __root_path__
from disdrodb.api.configs import available_sensor_names
from disdrodb.l0.check_standards import (
    _check_raw_fields_available,
    _check_valid_range,
    _check_valid_values,
    check_l0a_column_names,
    check_l0a_standards,
)
from disdrodb.l0.standards import get_raw_array_nvalues, get_sensor_logged_variables

BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")


def test_check_l0a_standards(capfd):
    filepath = os.path.join(
        BASE_DIR,
        "Raw",
        "EPFL",
        "PARSIVEL_2007",
        "ground_truth",
        "10",
        "L0A.PARSIVEL_2007.10.s20070723141530.e20070723141930.V0.parquet",
    )

    # Read apache parquet file and check that check pass
    df = pd.read_parquet(filepath)
    assert check_l0a_standards(df, sensor_name="OTT_Parsivel") is None

    # Now add longitude and latitude columns and check it logs info
    df["longitude"] = 1
    df["latitude"] = 1
    check_l0a_standards(df, sensor_name="OTT_Parsivel", verbose=True)
    # Capture the stdout
    out, _ = capfd.readouterr()
    assert "L0A dataframe has column 'latitude'" in out
    assert "L0A dataframe has column 'longitude'" in out


def test_check_valid_range():
    # Test case 1: All columns within range
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [0.5, 1.2, 2.7, 3.8]})
    dict_data_range = {"col1": [0, 5], "col2": [0, 4]}
    assert _check_valid_range(df, dict_data_range) is None

    # Test case 2: Some columns outside range
    df = pd.DataFrame({"col1": [1, 2, 10, 4], "col2": [0.5, 5.2, 2.7, 3.8]})
    dict_data_range = {"col1": [0, 5], "col2": [0, 4]}
    with pytest.raises(ValueError, match=r".*Columns \['col1', 'col2'\] has values outside the expected data range.*"):
        _check_valid_range(df, dict_data_range)

    # Test case 3: Empty dataframe
    df = pd.DataFrame()
    dict_data_range = {"col1": [0, 5], "col2": [0, 4]}
    assert _check_valid_range(df, dict_data_range) is None

    # Test case 4: Non-existing columns
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [0.5, 1.2, 2.7, 3.8]})
    dict_data_range = {"col1": [0, 5], "col3": [0, 4]}
    assert _check_valid_range(df, dict_data_range) is None


def test_check_valid_values():
    # Test case 1: All columns have valid values
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    dict_valid_values = {"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]}
    assert _check_valid_values(df, dict_valid_values) is None

    # Test case 2: Some columns have invalid values
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [1, 5, 3, 4]})
    dict_valid_values = {"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]}
    with pytest.raises(ValueError):
        _check_valid_values(df, dict_valid_values)

    # Test case 3: Empty dataframe
    df = pd.DataFrame()
    dict_valid_values = {"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]}
    assert _check_valid_values(df, dict_valid_values) is None

    # Test case 4: Non-existing columns
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    dict_valid_values = {"col1": [1, 2, 3, 4], "col3": [1, 2, 3, 4]}
    assert _check_valid_values(df, dict_valid_values) is None


def test_check_raw_fields_available():
    # Test case 1: Missing 'raw_drop_number' column
    df = pd.DataFrame({"other_column": [1, 2, 3]})
    sensor_name = "OTT_Parsivel"
    with pytest.raises(ValueError):
        _check_raw_fields_available(df, sensor_name)

    # Test case 2: All required columns present
    sensor_names = available_sensor_names(product="L0A")
    for sensor_name in sensor_names:
        n_bins_dict = get_raw_array_nvalues(sensor_name=sensor_name)
        raw_vars = np.array(list(n_bins_dict.keys()))
        dict_data = {i: [1, 2] for i in raw_vars}
        df = pd.DataFrame.from_dict(dict_data)
        assert _check_raw_fields_available(df, sensor_name) is None


def test_check_l0a_column_names(capsys):
    sensor_names = available_sensor_names(product="L0A")
    sensor_name = sensor_names[0]

    # Test 1 : All columns are present
    column_names = [*get_sensor_logged_variables(sensor_name), "time", "latitude", "longitude"]
    dict_data = {i: [1, 2] for i in column_names}
    df = pd.DataFrame.from_dict(dict_data)
    assert check_l0a_column_names(df, sensor_name=sensor_name) is None

    # Test 2 : Missing columns time
    column_names = [*get_sensor_logged_variables(sensor_name), "latitude", "longitude"]
    dict_data = {i: [1, 2] for i in column_names}
    df = pd.DataFrame.from_dict(dict_data)
    with pytest.raises(ValueError):
        check_l0a_column_names(df, sensor_name=sensor_name)

    # Test 3 : fake panda dataframe
    data = {"wrong_column_name": ["John", "Jane", "Bob", "Sara"]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        check_l0a_column_names(df, sensor_name=sensor_name)
