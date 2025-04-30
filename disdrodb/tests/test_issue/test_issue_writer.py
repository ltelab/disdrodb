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
"""Test DISDRODB issue writer."""
import os
from io import StringIO

import numpy as np
import pytest
import yaml

from disdrodb.issue.reader import read_issue, read_station_issue
from disdrodb.issue.writer import _write_issue_docs, create_station_issue, write_issue


def test_write_issue_docs():
    """Test the writing of the issue YAML documentation."""
    # Create a mock file object
    mock_file = StringIO()

    # Call the function under test
    _write_issue_docs(mock_file)

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


def test_write_issue(tmpdir):
    """Test the write_issue function."""
    # Define test inputs
    filepath = os.path.join(tmpdir, "test_yml")
    timesteps = np.array([0, 1, 2])
    time_periods = np.array([[0, 1], [2, 3]])

    # Call function
    write_issue(filepath, timesteps=timesteps, time_periods=time_periods)

    # Load YAML file
    with open(filepath) as f:
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

    write_issue(filepath, timesteps=np.array(timesteps), time_periods=None)

    result = read_issue(filepath)

    timesteps_datetime = np.array(timesteps, dtype="datetime64[s]")
    expected_result = {
        "timesteps": timesteps_datetime,
        "time_periods": None,
    }
    # assert np.array_equal(result,expected_result)
    assert set(result.keys()) == set(expected_result.keys())
    assert np.array_equal(result["timesteps"], expected_result["timesteps"])


def test_create_station_issue(tmp_path):
    """Test the creation of the default issue YAML file."""
    metadata_dir = tmp_path / "metadata" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    _ = create_station_issue(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    issue_dict = read_station_issue(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    assert isinstance(issue_dict, dict)
    issue_dict["timesteps"] = None
    issue_dict["time_periods"] = None

    # Test it raise error if creating when already existing
    with pytest.raises(ValueError):
        create_station_issue(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
