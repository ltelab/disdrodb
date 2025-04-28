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
"""Check DISDRODB L0 issues processing."""
import numpy as np
import pytest

from disdrodb.issue.checks import (
    _check_time_period_nested_list_format,
    _get_issue_time_periods,
    _get_issue_timesteps,
    _is_numpy_array_datetime,
    _is_numpy_array_string,
    check_issue_dict,
    check_time_periods,
    check_timesteps,
)

####--------------------------------------------------------------------------.
#### Checks


def test__is_numpy_array_string():
    # Test string array
    arr = np.array(["foo", "bar"], dtype=np.str_)
    assert _is_numpy_array_string(arr)

    # Test nonstring array
    arr = np.array([1, 2, 3])
    assert not _is_numpy_array_string(arr)

    # Test mixed type array
    arr = np.array(["foo", 1, 2.0], dtype=np.object_)
    assert not _is_numpy_array_string(arr)


####--------------------------------------------------------------------------.
#### Writer


def test__is_numpy_array_datetime():
    arr = np.array(["2022-01-01", "2022-01-02"], dtype="datetime64")
    assert _is_numpy_array_datetime(arr)

    arr = np.array([1, 2, 3])
    assert not _is_numpy_array_datetime(arr)


def test_check_timesteps():
    """Check validity testing of timesteps."""
    # Test None input
    assert check_timesteps(None) is None

    # Test correct string input
    timesteps_string = "2022-01-01 01:00:00"
    expected_output_string = np.array(["2022-01-01T01:00:00"], dtype="datetime64[s]")
    assert np.array_equal(check_timesteps(timesteps_string), expected_output_string)

    # Test correct list of string inputs
    timesteps_string_list = ["2022-01-01 01:00:00", "2022-01-01 02:00:00"]
    expected_output_string_list = np.array(["2022-01-01T01:00:00", "2022-01-01T02:00:00"], dtype="datetime64[s]")
    assert np.array_equal(check_timesteps(timesteps_string_list), expected_output_string_list)

    # Test correct datetime input
    timesteps_datetime = np.array(["2022-01-01T01:00:00", "2022-01-01T02:00:00"], dtype="datetime64[s]")
    expected_output_datetime = np.array(["2022-01-01T01:00:00", "2022-01-01T02:00:00"], dtype="datetime64[s]")
    assert np.array_equal(check_timesteps(timesteps_datetime), expected_output_datetime)

    # Test invalid type input
    with pytest.raises(TypeError):
        check_timesteps(123)

    # Test invalid datetime input
    timesteps = np.array(["2022-01-01", "2022-01-02"], dtype="datetime64[D]")
    with pytest.raises(ValueError):
        check_timesteps(timesteps)

    # Test invalid list of string (wrong temporal resolution)
    timesteps = ["2022-01-01 01:00", "2022-01-01 02:00"]
    with pytest.raises(ValueError):
        check_timesteps(timesteps)

    # Test invalid list of string (wrong time format)
    timesteps = ["2022-15-01 01:00:00", "2022-15-01 02:00:00"]
    with pytest.raises(ValueError):
        check_timesteps(timesteps)


def test_check_time_period_nested_list_format():
    # Test valid input
    time_periods_valid = [
        ["2022-01-01 01:00:00", "2022-01-01 02:00:00"],
        ["2022-01-02 01:00:00", "2022-01-02 02:00:00"],
    ]
    assert _check_time_period_nested_list_format(time_periods_valid) is None

    # Test invalid input type
    time_periods_invalid_type = "not a list"
    with pytest.raises(TypeError):
        _check_time_period_nested_list_format(time_periods_invalid_type)

    # Test invalid input length
    time_periods_invalid_length = [["2022-01-01 01:00:00", "2022-01-01 02:00:00"], ["2022-01-02 01:00:00"]]
    with pytest.raises(ValueError):
        _check_time_period_nested_list_format(time_periods_invalid_length)

    # Test invalid input element type
    time_periods_invalid_element_type = [["2022-01-01 01:00:00", 123], ["2022-01-02 01:00:00", "2022-01-02 02:00:00"]]
    assert _check_time_period_nested_list_format(time_periods_invalid_element_type) is None


def test_check_time_periods():
    # Valid input
    time_periods = [
        ["2022-01-01 01:00:00", "2022-01-01 02:00:00"],
        ["2022-01-02 01:00:00", "2022-01-02 02:00:00"],
    ]

    expected_result = [np.array(time_period, dtype="datetime64[s]") for time_period in time_periods]
    assert np.array_equal(check_time_periods(time_periods), expected_result)

    # None input
    assert check_time_periods(None) is None

    # Invalid input: invalid time period
    time_periods = [
        ["2022-01-01 01:00:00", "2022-01-01 02:00:00"],
        ["2022-01-02 01:00:00", "2021-01-02 02:00:00"],
    ]
    with pytest.raises(ValueError):
        check_time_periods(time_periods)

    # Invalid input: invalid format
    time_periods = [
        ["2022-01-01 01:00:00", "2022-01-01 02:00:00"],
        ["2022-01-02 01:00:00"],
    ]
    with pytest.raises(ValueError):
        check_time_periods(time_periods)


def test_get_issue_timesteps():
    # Test case 1: Valid timesteps
    time_periods = ["2022-01-01 01:00:00", "2022-01-01 02:00:00"]
    issue_dict = {"timesteps": time_periods}
    expected_result = [np.array(time_period, dtype="datetime64[s]") for time_period in time_periods]
    assert np.array_equal(_get_issue_timesteps(issue_dict), expected_result)


def test__get_issue_time_periods():
    # Test case 1: Valid time periods
    time_periods = [["2022-01-01 01:00:00", "2022-01-01 02:00:00"], ["2022-01-02 01:00:00", "2022-01-02 02:00:00"]]
    issue_dict = {"time_periods": time_periods}
    expected_result = [np.array(time_period, dtype="datetime64[s]") for time_period in time_periods]
    assert np.array_equal(_get_issue_time_periods(issue_dict), expected_result)

    # Test case 2: No time periods
    issue_dict = {}
    assert _get_issue_time_periods(issue_dict) is None


def test_check_issue_dict():
    # Test empty dictionary
    assert check_issue_dict({}) == {}

    # Test dictionary with invalid keys
    # with pytest.raises(ValueError):
    #     check_issue_dict({"foo": "bar"})

    # Test dictionary with valid keys and timesteps
    timesteps = ["2022-01-01 01:00:00", "2022-01-01 02:00:00"]

    issue_dict = {
        "timesteps": timesteps,
    }

    timesteps_datetime = np.array(timesteps, dtype="datetime64[s]")
    result = check_issue_dict(issue_dict)
    expected_result = {
        "timesteps": timesteps_datetime,
        "time_periods": None,
    }

    assert set(result.keys()) == set(expected_result.keys())

    # Test timesteps kees
    assert np.array_equal(result["timesteps"], expected_result["timesteps"])

    # Test invalid keys
    issue_dict = {"timesteps": timesteps, "invalid_key": "invalid_value"}
    with pytest.raises(ValueError):
        check_issue_dict(issue_dict)
