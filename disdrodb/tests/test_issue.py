import os
import numpy as np
from io import StringIO
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


# def test_write_issue():
#     # Create a mock file path
#     mock_file_path = os.path.join(
#         PATH_TEST_FOLDERS_FILES, "test_folders_files_creation", "mock_file_path.yml"
#     )

#     # Create a mock timestamp
#     mock_timesteps = np.array(["2018-12-07 14:15:00"])

#     # Create a mock time period
#     mock_time_periods = [np.array(["2018-08-01 12:00:00", "2018-08-01 14:00:00"])]

#     # Call the function under test with the mock input
#     issue._write_issue(fpath=mock_file_path,
#                        timesteps=mock_timesteps,
#                        time_periods=mock_time_periods)

#     # Read the created YAML file and check that it contains the expected data
#     issue_dict = issue.load_yaml_without_date_parsing(mock_file_path)

#     # Check that the data in the YAML file matches the expected output
#     assert issue_dict["timesteps"] == mock_timesteps.tolist()
#     assert issue_dict["time_periods"] == mock_time_periods.tolist()

#     # Delete the mock file
#     os.remove(mock_file_path)


####---------------------------------------------------------------------------.
#### Reader
# def test_read_issue():
#     # Create a mock file path
#     raw_dir = os.path.join(PATH_TEST_FOLDERS_FILES, "test_folders_files_creation")
#     station_name = "123"
#     issue_dir = os.path.join(raw_dir, "issue")

#     if not os.path.exists(issue_dir):
#         os.makedirs(issue_dir)

#     issue_fpath = os.path.join(issue_dir, station_name + ".yml")
#     with open(issue_fpath, "w") as f:
#         yaml.safe_dump({"key": "value"}, f)

#     # Ensure the read_issue function returns the correct output
#     assert issue.read_issue(raw_dir, station_name) == {"key": "value"}
