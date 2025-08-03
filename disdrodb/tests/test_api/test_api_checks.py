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
"""Test DISDRODB API checks utility."""
import datetime
import os

import numpy as np
import pandas as pd
import pytest
import pytz

from disdrodb import __root_path__
from disdrodb.api.checks import (
    check_data_archive_dir,
    check_filepaths,
    check_path,
    check_path_is_a_directory,
    check_sensor_name,
    check_start_end_time,
    check_time,
    check_url,
    get_current_utc_time,
)

TEST_DATA_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data")


def test_check_path():
    # Test a valid path
    path = os.path.abspath(__file__)
    assert check_path(path) is None

    # Test an invalid path
    path = "/path/that/does/not/exist"
    with pytest.raises(FileNotFoundError):
        check_path(path)


def test_check_url():
    # Test with valid URLs
    assert check_url("https://www.example.com")
    assert check_url("http://example.com/path/to/file.html?param=value")
    assert check_url("www.example.com")
    assert check_url("example.com")

    # Test with invalid URLs
    assert not check_url("ftp://example.com")
    assert not check_url("htp://example.com")
    assert not check_url("http://example.com/path with spaces")


def test_check_data_archive_dir():
    from pathlib import Path

    data_archive_dir = os.path.join("path", "to", "DISDRODB")
    assert check_data_archive_dir(data_archive_dir) == data_archive_dir

    assert check_data_archive_dir(Path(data_archive_dir)) == data_archive_dir

    with pytest.raises(ValueError):
        check_data_archive_dir("/path/to/DISDRO")


def test_check_sensor_name():
    sensor_name = "wrong_sensor_name"

    # Test with an unknown device
    with pytest.raises(ValueError):
        check_sensor_name(sensor_name)

    # Test with a woronf type
    with pytest.raises(TypeError):
        check_sensor_name(123)


def test_check_path_is_a_directory(tmp_path):
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    data_archive_dir.mkdir(parents=True, exist_ok=True)
    check_path_is_a_directory(str(data_archive_dir))
    check_path_is_a_directory(data_archive_dir)


def test_check_filepaths() -> None:
    """Check path constructor for filepaths."""
    # Create list of unique filepaths (may not reflect real files)
    filepaths = [
        os.path.join("dummy", "path"),
        os.path.join("dummy", "path1"),
    ]

    res = check_filepaths(filepaths)
    assert res == filepaths, "List of filepaths is not returned"

    # Check if single string is converted to list
    res = check_filepaths(filepaths[0])
    assert res == [filepaths[0]], "String is not converted to list"

    # Check if not list or string, TypeError is raised
    with pytest.raises(TypeError):
        check_filepaths(123)


def test_check_time() -> None:
    """Test that time is returned a `datetime.datetime` object from varying inputs."""
    # Test a string
    res = check_time("2014-12-31")
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a string with hh/mm/ss
    res = check_time("2014-12-31 12:30:30")
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a string with <date>T<time>
    res = check_time("2014-12-31T12:30:30")
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a datetime object
    res = check_time(datetime.datetime(2014, 12, 31))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a datetime timestamp with h/m/s/ms
    res = check_time(datetime.datetime(2014, 12, 31, 12, 30, 30, 300))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30, 300)

    # Test a numpy.datetime64 object of "datetime64[s]"
    res = check_time(np.datetime64("2014-12-31"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a object of datetime64[ns] casts to datetime64[ms]
    res = check_time(np.datetime64("2014-12-31T12:30:30.934549845", "s"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a datetime.date
    res = check_time(datetime.date(2014, 12, 31))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a datetime object inside a numpy array
    with pytest.raises(ValueError):
        res = check_time(np.array([datetime.datetime(2014, 12, 31, 12, 30, 30)]))

    # Test a pandas Timestamp object inside a numpy array
    with pytest.raises(ValueError):
        res = check_time(np.array([pd.Timestamp("2014-12-31 12:30:30")]))

    # Test a pandas Timestamp object
    res = check_time(pd.Timestamp("2014-12-31 12:30:30"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test automatic casting to seconds accuracy
    res = check_time(np.datetime64("2014-12-31T12:30:30.934549845", "ns"))
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a non isoformat string
    with pytest.raises(ValueError):
        check_time("2014/12/31")

    # Test a non datetime object
    with pytest.raises(TypeError):
        check_time(123)

    # Check numpy single timestamp
    res = check_time(np.array(["2014-12-31"], dtype="datetime64[s]"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Check numpy multiple timestamp
    with pytest.raises(ValueError):
        check_time(np.array(["2014-12-31", "2015-01-01"], dtype="datetime64[s]"))

    # Test with numpy non datetime64 object
    with pytest.raises(ValueError):
        check_time(np.array(["2014-12-31"]))

    # Check non-UTC timezone
    with pytest.raises(ValueError):
        check_time(
            datetime.datetime(2014, 12, 31, 12, 30, 30, 300, tzinfo=pytz.timezone("Europe/Zurich")),
        )


def test_check_start_end_time() -> None:
    """Check start and end time are valid."""
    # Test a string
    res = check_start_end_time(
        "2014-12-31",
        "2015-01-01",
    )
    assert isinstance(res, tuple)

    # Test the reverse for exception
    with pytest.raises(ValueError):
        check_start_end_time(
            "2015-01-01",
            "2014-12-31",
        )

    # Test a datetime object
    res = check_start_end_time(
        datetime.datetime(2014, 12, 31),
        datetime.datetime(2015, 1, 1),
    )
    assert isinstance(res, tuple)

    # Test the reverse datetime object for exception
    with pytest.raises(ValueError):
        check_start_end_time(
            datetime.datetime(2015, 1, 1),
            datetime.datetime(2014, 12, 31),
        )

    # Test a datetime timestamp with h/m/s/ms
    res = check_start_end_time(
        datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
        datetime.datetime(2015, 1, 1, 12, 30, 30, 300),
    )

    # Test end time in the future
    with pytest.raises(ValueError):
        check_start_end_time(
            datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
            datetime.datetime(2125, 1, 1, 12, 30, 30, 300),
        )

    # Test start time in the future
    with pytest.raises(ValueError):
        check_start_end_time(
            datetime.datetime(2125, 12, 31, 12, 30, 30, 300),
            datetime.datetime(2126, 1, 1, 12, 30, 30, 300),
        )

    # Check that a timestep generated now in another timezone with no tzinfo, throw error
    for timezone in ["Europe/Zurich", "Australia/Melbourne"]:
        with pytest.raises(ValueError):
            check_start_end_time(
                datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
                datetime.datetime.now(tz=pytz.timezone(timezone)).replace(tzinfo=None),
            )

    # Specifying timezone different than UTC should throw exception
    for timezone in ["Europe/Zurich", "Australia/Melbourne"]:
        with pytest.raises(ValueError):
            check_start_end_time(
                datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
                datetime.datetime.now(tz=pytz.timezone(timezone)),
            )

    # This should pass as the time is in UTC
    check_start_end_time(
        datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
        datetime.datetime.now(tz=pytz.utc),
    )

    # Do the same but in a timezone that is behind UTC (this should pass)
    for timezone in ["America/New_York", "America/Santiago"]:
        check_start_end_time(
            datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
            datetime.datetime.now(tz=pytz.timezone(timezone)).replace(tzinfo=None),
        )

    # Test endtime in UTC. This should pass as UTC time generated in the test is slightly
    # behind the current time tested in the function
    check_start_end_time(
        datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
        get_current_utc_time(),
    )
