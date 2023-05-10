import os
from io import StringIO

import numpy as np
import pytest
import yaml

from disdrodb.l0 import issue

PATH_TEST_FOLDERS_FILES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytest_files")

####--------------------------------------------------------------------------.
#### Checks


def test_is_numpy_array_string():
    # Test string array
    arr = np.array(["foo", "bar"], dtype=np.str_)
    assert issue.is_numpy_array_string(arr) is True

    # Test unicode array
    arr = np.array(["foo", "bar"], dtype=np.unicode_)
    assert issue.is_numpy_array_string(arr) is True

    # Test nonstring array
    arr = np.array([1, 2, 3])
    assert issue.is_numpy_array_string(arr) is False

    # Test mixed type array
    arr = np.array(["foo", 1, 2.0], dtype=np.object_)
    assert issue.is_numpy_array_string(arr) is False


def test_check_issue_file():
    # function_return = issue.check_issue_file()
    assert 1 == 1


####--------------------------------------------------------------------------.
#### Writer


def test_write_issue_docs():
    # Create a mock file object
    mock_file = StringIO()

    # Call the function under test
    issue._write_issue_docs(mock_file)

    # Get the written data from the mock file object
    written_data = mock_file.getvalue()

    # Check that the written data matches the expected output
    expected_output = """# This file is used to store timesteps/time periods with wrong/corrupted observation.
# The specified timesteps are dropped during the L0 processing.
# The time format used is the isoformat : YYYY-mm-dd HH:MM:SS.
# The 'timesteps' key enable to specify the list of timesteps to be discarded.
# The 'time_period' key enable to specify the time periods to be dropped.
# Example:
#
# timesteps:
# - 2018-12-07 14:15:00
# - 2018-12-07 14:17:00
# - 2018-12-07 14:19:00
# - 2018-12-07 14:25:00
# time_period:
# - ['2018-08-01 12:00:00', '2018-08-01 14:00:00']
# - ['2018-08-01 15:44:30', '2018-08-01 15:59:31']
# - ['2018-08-02 12:44:30', '2018-08-02 12:59:31'] \n
"""
    assert written_data == expected_output


def test_is_numpy_array_datetime():
    arr = np.array(["2022-01-01", "2022-01-02"], dtype="datetime64")
    assert issue.is_numpy_array_datetime(arr) is True

    arr = np.array([1, 2, 3])
    assert issue.is_numpy_array_datetime(arr) is False


def test__check_timestep_datetime_accuracy():
    timesteps = np.array(["2022-01-01T01:00:00", "2022-01-01T02:00:00"], dtype="datetime64[s]")
    assert np.array_equal(issue._check_timestep_datetime_accuracy(timesteps, unit="s"), timesteps)

    with pytest.raises(ValueError):
        timesteps = np.array(["2022-01-01", "2022-01-02"], dtype="datetime64[D]")
        issue._check_timestep_datetime_accuracy(timesteps, unit="s")


def test__check_timesteps_string():
    timesteps = ["2022-01-01 01:00:00", "2022-01-01 02:00:00"]
    expected_output = np.array(["2022-01-01T01:00:00", "2022-01-01T02:00:00"], dtype="datetime64[s]")
    assert np.array_equal(issue._check_timesteps_string(timesteps), expected_output)

    with pytest.raises(ValueError):
        timesteps = ["2022-01-01 01:00", "2022-01-01 02:00:00"]
        issue._check_timesteps_string(timesteps)


def test_check_timesteps():
    # Test None input
    assert issue.check_timesteps(None) is None

    # Test string input
    timesteps_string = "2022-01-01 01:00:00"
    expected_output_string = np.array(["2022-01-01T01:00:00"], dtype="datetime64[s]")
    assert np.array_equal(issue.check_timesteps(timesteps_string), expected_output_string)

    # Test list of string inputs
    timesteps_string_list = ["2022-01-01 01:00:00", "2022-01-01 02:00:00"]
    expected_output_string_list = np.array(["2022-01-01T01:00:00", "2022-01-01T02:00:00"], dtype="datetime64[s]")
    assert np.array_equal(issue.check_timesteps(timesteps_string_list), expected_output_string_list)

    # Test datetime input
    timesteps_datetime = np.array(["2022-01-01T01:00:00", "2022-01-01T02:00:00"], dtype="datetime64[s]")
    expected_output_datetime = np.array(["2022-01-01T01:00:00", "2022-01-01T02:00:00"], dtype="datetime64[s]")
    assert np.array_equal(issue.check_timesteps(timesteps_datetime), expected_output_datetime)

    # Test invalid input
    with pytest.raises(TypeError):
        issue.check_timesteps(123)


def test_check_time_period_nested_list_format():
    # Test valid input
    time_periods_valid = [
        ["2022-01-01 01:00:00", "2022-01-01 02:00:00"],
        ["2022-01-02 01:00:00", "2022-01-02 02:00:00"],
    ]
    assert issue._check_time_period_nested_list_format(time_periods_valid) is None

    # Test invalid input type
    time_periods_invalid_type = "not a list"
    with pytest.raises(TypeError):
        issue._check_time_period_nested_list_format(time_periods_invalid_type)

    # Test invalid input length
    time_periods_invalid_length = [["2022-01-01 01:00:00", "2022-01-01 02:00:00"], ["2022-01-02 01:00:00"]]
    with pytest.raises(ValueError):
        issue._check_time_period_nested_list_format(time_periods_invalid_length)

    # Test invalid input element type
    time_periods_invalid_element_type = [["2022-01-01 01:00:00", 123], ["2022-01-02 01:00:00", "2022-01-02 02:00:00"]]
    assert issue._check_time_period_nested_list_format(time_periods_invalid_element_type) is None


def test_check_time_periods():
    # Valid input
    time_periods = [
        ["2022-01-01 01:00:00", "2022-01-01 02:00:00"],
        ["2022-01-02 01:00:00", "2022-01-02 02:00:00"],
    ]

    expected_result = [np.array(time_period, dtype="datetime64[s]") for time_period in time_periods]
    assert np.array_equal(issue.check_time_periods(time_periods), expected_result)

    # None input
    assert issue.check_time_periods(None) is None

    # Invalid input: unvalid time period
    time_periods = [
        ["2022-01-01 01:00:00", "2022-01-01 02:00:00"],
        ["2022-01-02 01:00:00", "2021-01-02 02:00:00"],
    ]
    with pytest.raises(ValueError):
        issue.check_time_periods(time_periods)

    # Invalid input: invalid format
    time_periods = [
        ["2022-01-01 01:00:00", "2022-01-01 02:00:00"],
        ["2022-01-02 01:00:00"],
    ]
    with pytest.raises(ValueError):
        issue.check_time_periods(time_periods)


def test_get_issue_timesteps():
    # Test case 1: Valid timesteps
    time_periods = ["2022-01-01 01:00:00", "2022-01-01 02:00:00"]
    issue_dict = {"timesteps": time_periods}
    expected_result = [np.array(time_period, dtype="datetime64[s]") for time_period in time_periods]
    assert np.array_equal(issue._get_issue_timesteps(issue_dict), expected_result)


def test__get_issue_time_periods():
    # Test case 1: Valid time periods
    time_periods = [["2022-01-01 01:00:00", "2022-01-01 02:00:00"], ["2022-01-02 01:00:00", "2022-01-02 02:00:00"]]
    issue_dict = {"time_periods": time_periods}
    expected_result = [np.array(time_period, dtype="datetime64[s]") for time_period in time_periods]
    assert np.array_equal(issue._get_issue_time_periods(issue_dict), expected_result)

    # Test case 2: No time periods
    issue_dict = {}
    assert issue._get_issue_time_periods(issue_dict) is None


def test_check_issue_dict():
    # Test empty dictionary
    assert issue.check_issue_dict({}) == {}

    # Test dictionary with invalid keys
    # with pytest.raises(ValueError):
    #     issue.check_issue_dict({"foo": "bar"})

    # Test dictionary with valid keys and timesteps
    timesteps = ["2022-01-01 01:00:00", "2022-01-01 02:00:00"]

    issue_dict = {
        "timesteps": timesteps,
    }

    timesteps_datetime = np.array(timesteps, dtype="datetime64[s]")
    result = issue.check_issue_dict(issue_dict)
    expected_result = {
        "timesteps": timesteps_datetime,
        "time_periods": None,
    }

    assert set(result.keys()) == set(expected_result.keys())

    # Test timesteps kees
    assert np.array_equal(result["timesteps"], expected_result["timesteps"])

    # Test unvalid keys
    issue_dict = {"timesteps": timesteps, "unvalid_key": "unvalid_value"}
    with pytest.raises(ValueError):
        issue.check_issue_dict(issue_dict)


def test_write_issue(tmpdir):
    """Test the _write_issue function."""
    # Define test inputs
    fpath = os.path.join(tmpdir, "test_issue.yml")
    timesteps = np.array([0, 1, 2])
    time_periods = np.array([[0, 1], [2, 3]])

    # Call function
    issue._write_issue(fpath, timesteps=timesteps, time_periods=time_periods)

    # Load YAML file
    with open(fpath, "r") as f:
        issue_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Check the issue dictionary
    assert isinstance(issue_dict, dict)
    assert len(issue_dict) == 2
    assert issue_dict.keys() == {"timesteps", "time_periods"}
    assert np.array_equal(issue_dict["timesteps"], timesteps.astype(str).tolist())
    assert np.array_equal(issue_dict["time_periods"], time_periods.astype(str).tolist())

    # Test dictionary with valid keys and timesteps
    timesteps = ["2022-01-01 01:00:00", "2022-01-01 02:00:00"]

    issue_dict = {
        "timesteps": timesteps,
    }

    issue._write_issue(fpath, timesteps=np.array(timesteps), time_periods=None)

    result = issue.read_issue_file(fpath)

    timesteps_datetime = np.array(timesteps, dtype="datetime64[s]")
    expected_result = {
        "timesteps": timesteps_datetime,
        "time_periods": None,
    }
    # assert np.array_equal(result,expected_result)
    assert set(result.keys()) == set(expected_result.keys())
    assert np.array_equal(result["timesteps"], expected_result["timesteps"])
