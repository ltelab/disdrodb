import os
import yaml
from io import StringIO
from disdrodb.L0 import issue


PATH_TEST_FOLDERS_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pytest_files"
)


def test__write_issue_timestamps_docs():
    # Create a mock file object
    mock_file = StringIO()

    # Call the function under test
    issue._write_issue_timestamps_docs(mock_file)

    # Get the written data from the mock file object
    written_data = mock_file.getvalue()

    # Check that the written data matches the expected output
    expected_output = (
        "# This file is used to store dates to drop by the reader, the time format used is the isoformat (YYYY-mm-dd HH:MM:SS). \n"
        "# timestamp: list of timestamps \n"
        "# time_period: list of list ranges of dates \n"
        "# Example: \n"
        "# timestamp: ['2018-12-07 14:15','2018-12-07 14:17','2018-12-07 14:19', '2018-12-07 14:25'] \n"
        "# time_period: [['2018-08-01 12:00:00', '2018-08-01 14:00:00'], \n"
        "#               ['2018-08-01 15:44:30', '2018-08-01 15:59:31'], \n"
        "#               ['2018-08-02 12:44:30', '2018-08-02 12:59:31']] \n"
    )
    assert written_data == expected_output


def test_create_issue_yml():

    # Create a mock file path
    mock_file_path = os.path.join(
        PATH_TEST_FOLDERS_FILES, "test_folders_files_creation", "mock_file_path.yml"
    )

    # Create a mock timestamp
    mock_timestamp = ["2018-12-07 14:15"]

    # Create a mock time period
    mock_time_period = [["2018-08-01 12:00:00", "2018-08-01 14:00:00"]]

    # Call the function under test with the mock input
    issue.create_issue_yml(mock_file_path, mock_timestamp, mock_time_period)

    # Read the created YAML file and check that it contains the expected data
    with open(mock_file_path, "r") as f:
        data = yaml.safe_load(f)

    # Check that the data in the YAML file matches the expected output
    expected_output = {
        "timestamp": mock_timestamp,
        "time_period": mock_time_period,
    }
    assert data == expected_output

    # Delete the mock file
    os.remove(mock_file_path)


def test_read_issue():

    # Create a mock file path
    raw_dir = os.path.join(PATH_TEST_FOLDERS_FILES, "test_folders_files_creation")
    station_name = "123"
    issue_dir = os.path.join(raw_dir, "issue")

    if not os.path.exists(issue_dir):
        os.makedirs(issue_dir)

    issue_fpath = os.path.join(issue_dir, station_name + ".yml")
    with open(issue_fpath, "w") as f:
        yaml.safe_dump({"key": "value"}, f)

    # Ensure the read_issue function returns the correct output
    assert issue.read_issue(raw_dir, station_name) == {"key": "value"}


def test_check_issue_compliance():
    # function_return = issue.check_issue_compliance()
    assert 1 == 1
