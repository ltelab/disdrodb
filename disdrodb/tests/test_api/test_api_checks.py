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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

from disdrodb import __root_path__
from disdrodb.api.checks import (
    check_campaign_name,
    check_campaign_names,
    check_data_archive_dir,
    check_data_availability,
    check_data_source,
    check_data_sources,
    check_directories_inside,
    check_filepaths,
    check_folder_partitioning,
    check_invalid_fields_policy,
    check_issue_dir,
    check_issue_file,
    check_measurement_interval,
    check_measurement_intervals,
    check_metadata_archive_dir,
    check_metadata_file,
    check_path,
    check_path_is_a_directory,
    check_product,
    check_product_kwargs,
    check_rolling,
    check_sample_interval,
    check_scattering_table_dir,
    check_sensor_name,
    check_start_end_time,
    check_station_inputs,
    check_station_names,
    check_time,
    check_url,
    check_valid_fields,
    get_current_utc_time,
    has_available_data,
    select_required_product_kwargs,
)
from disdrodb.api.path import define_data_dir, define_issue_dir, define_issue_filepath
from disdrodb.constants import PRODUCTS, PRODUCTS_ARGUMENTS

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


class TestCheckPathIsADirectory:
    """Tests for check_path_is_a_directory function."""

    def test_valid_directory(self, tmp_path):
        """Test passes with an existing directory (string and Path)."""
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_archive_dir.mkdir(parents=True, exist_ok=True)
        check_path_is_a_directory(str(data_archive_dir))
        check_path_is_a_directory(data_archive_dir)

    def test_non_existing_directory(self, tmp_path):
        """Test raises ValueError when directory does not exist."""
        missing_dir = tmp_path / "missing"
        with pytest.raises(ValueError):
            check_path_is_a_directory(missing_dir)

    def test_path_is_a_file(self, tmp_path):
        """Test raises ValueError when path is a file, not a directory."""
        f = tmp_path / "file.txt"
        f.write_text("dummy")
        with pytest.raises(ValueError):
            check_path_is_a_directory(f)


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


def test_check_directories_inside(tmp_path):
    """Test check_directories_inside raises if no subdirectories exist."""
    with pytest.raises(ValueError):
        check_directories_inside(tmp_path)

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    check_directories_inside(tmp_path)  # should pass


def test_check_metadata_archive_dir():
    """Test check_metadata_archive_dir enforces 'DISDRODB' ending."""
    path = os.path.join("some", "DISDRODB")
    assert check_metadata_archive_dir(path) == os.path.normpath(path)

    with pytest.raises(ValueError):
        check_metadata_archive_dir("wrong_path")


def test_check_scattering_table_dir(tmp_path):
    """Test check_scattering_table_dir verifies directory existence."""
    with pytest.raises(ValueError):
        check_scattering_table_dir(tmp_path / "missing")

    assert check_scattering_table_dir(tmp_path) == str(tmp_path)


def test_check_sample_interval_and_rolling():
    """Test check_sample_interval validation."""
    check_sample_interval(1)

    with pytest.raises(TypeError):
        check_sample_interval(3.5)

    with pytest.raises(TypeError):
        check_sample_interval("3.5")

    with pytest.raises(TypeError):
        check_sample_interval(True)


def test_check_rolling():
    """Test check_rolling validation."""
    check_rolling(True)

    with pytest.raises(TypeError):
        check_rolling("no")

    with pytest.raises(TypeError):
        check_rolling("1")

    with pytest.raises(TypeError):
        check_rolling(1)


def test_check_folder_partitioning():
    """Test check_folder_partitioning with valid and invalid schemes."""
    valid = ["", "year", "year/month", "year/month/day", "year/month_name", "year/quarter"]
    for v in valid:
        assert check_folder_partitioning(v) == v

    with pytest.raises(ValueError):
        check_folder_partitioning("month/year")


def test_check_data_source():
    """Test check_data_source enforce uppercase."""
    with pytest.raises(ValueError):
        check_data_source("source")

    check_data_source("UPPER")


def test_check_campaign_name():
    """Test check_campaign_name enforce uppercase."""
    with pytest.raises(ValueError):
        check_campaign_name("lowercase")

    check_campaign_name("UPPER")


def test_check_product():
    """Test check_product validates against PRODUCTS."""
    assert check_product(PRODUCTS[0]) == PRODUCTS[0]

    with pytest.raises(TypeError):
        check_product(123)

    with pytest.raises(ValueError):
        check_product("WRONG")


@pytest.mark.parametrize("product", list(PRODUCTS))
def test_check_product_kwargs(product):
    """Test check_product_kwargs for all products against required/extra args."""
    required_args = PRODUCTS_ARGUMENTS.get(product, [])

    # Case 1: valid kwargs (all required provided, no extras)
    kwargs = dict.fromkeys(required_args, 1)
    assert check_product_kwargs(product, kwargs) == kwargs

    # Case 2: missing required arguments
    if required_args:  # only makes sense if product has requirements
        with pytest.raises(ValueError):
            check_product_kwargs(product, {})

    # Case 3: extra argument
    with pytest.raises(ValueError):
        check_product_kwargs(product, {**kwargs, "extra": 1})

    # Case 4: missing and extra arguments
    with pytest.raises(ValueError):
        check_product_kwargs(product, {"extra": 1})


def test_select_required_product_kwargs():
    """Test select_required_product_kwargs returns required args or raises."""
    product = next(iter(PRODUCTS_ARGUMENTS.keys()))
    required_args = PRODUCTS_ARGUMENTS[product]
    kwargs = dict.fromkeys(required_args, 1)
    assert select_required_product_kwargs(product, kwargs) == kwargs
    if required_args:
        with pytest.raises(ValueError):
            select_required_product_kwargs(product, {})


def test_check_data_sources():
    """Test check_data_sources delegates to _check_fields."""
    assert check_data_sources(None) is None
    assert check_data_sources("a") == ["a"]
    assert sorted(check_data_sources(["a", "b", "a"])) == ["a", "b"]


def test_check_campaign_names():
    """Test check_campaign_names delegates to _check_fields."""
    assert check_campaign_names(None) is None
    assert check_campaign_names("a") == ["a"]
    assert sorted(check_campaign_names(["a", "b", "a"])) == ["a", "b"]


def test_check_station_names():
    """Test check_station_names delegates to _check_fields."""
    assert check_station_names(None) is None
    assert check_station_names("a") == ["a"]
    assert sorted(check_station_names(["a", "b", "a"])) == ["a", "b"]


def test_check_invalid_fields_policy():
    """Test check_invalid_fields_policy accepts only raise/warn/ignore."""
    for v in ["raise", "warn", "ignore"]:
        assert check_invalid_fields_policy(v) == v

    with pytest.raises(ValueError):
        check_invalid_fields_policy("invalid")


class TestCheckValidFields:
    """Test check_valid_fields under different policies."""

    def test_valid_fields(self):
        """Test valid fields are returned unchanged."""
        fields = ["a", "b"]
        available_fields = ["a", "b", "c"]
        field_name = "field"

        res = check_valid_fields(fields=fields, available_fields=available_fields, field_name=field_name)
        assert sorted(res) == ["a", "b"]

        # Test passing a string return a list
        res = check_valid_fields(fields="a", available_fields=available_fields, field_name=field_name)
        assert res == ["a"]

        # Test None returns None
        res = check_valid_fields(fields=None, available_fields=available_fields, field_name=field_name)
        assert res is None

    def test_invalid_fields_raise(self):
        """Test invalid fields raise ValueError under 'raise' policy."""
        fields = ["x", "a"]
        available_fields = ["a"]
        field_name = "field"

        with pytest.raises(ValueError):
            check_valid_fields(
                fields=fields,
                available_fields=available_fields,
                field_name=field_name,
                invalid_fields_policy="raise",
            )

    def test_invalid_fields_warn(self):
        """Test presence of some invalid fields trigger warnings under 'warn' policy."""
        fields = ["ab", "a"]
        available_fields = ["a"]
        field_name = "field"

        with pytest.warns(UserWarning):
            res = check_valid_fields(
                fields=fields,
                available_fields=available_fields,
                field_name=field_name,
                invalid_fields_policy="warn",
            )
        assert res == ["a"]

    def test_invalid_fields_ignore(self):
        """Test invalid fields are dropped under 'ignore' policy."""
        with pytest.raises(ValueError):
            # all invalid, results in empty list -> error
            check_valid_fields("x", ["a"], "field", invalid_fields_policy="ignore")

    def test_only_invalid_fields_raise_error(self):
        """Test that presence of only invalid fields trigger an error under 'warn' policy."""
        fields = ["x", "y"]
        available_fields = ["a"]
        field_name = "field"

        with pytest.raises(ValueError):
            check_valid_fields(
                fields=fields,
                available_fields=available_fields,
                field_name=field_name,
                invalid_fields_policy="warn",
            )

        with pytest.raises(ValueError):
            check_valid_fields(
                fields=fields,
                available_fields=available_fields,
                field_name=field_name,
                invalid_fields_policy="ignore",
            )


class TestMeasurementIntervals:
    """Test check_measurement_interval and check_measurement_intervals."""

    def test_valid_intervals(self):
        """Test integers and strings convert to int list."""
        assert check_measurement_interval(10) == 10
        assert check_measurement_interval("5") == 5
        assert check_measurement_intervals(10) == [10]
        assert check_measurement_intervals([5, "6"]) == [5, 6]

    def test_invalid_intervals(self):
        """Test empty string, None, and non-digits raise errors."""
        with pytest.raises(ValueError):
            check_measurement_interval("")
        with pytest.raises(ValueError):
            check_measurement_interval(None)
        with pytest.raises(ValueError):
            check_measurement_interval("abc")


class TestCheckStationInputs:

    def test_valid_station_inputs(self, disdrodb_metadata_archive_dir):
        """Test no error is raised with valid station inputs."""
        check_station_inputs(
            # DISDRODB root directories
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            # Station arguments
            data_source="EPFL",
            campaign_name="LOCARNO_2019",
            station_name="61",
        )

    def test_invalid_data_sources(self, disdrodb_metadata_archive_dir):
        """Test error is raised with invalid data source."""
        with pytest.raises(ValueError):
            check_station_inputs(
                # DISDRODB root directories
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                # Station arguments
                data_source="EPFL_WRONG",
                campaign_name="LOCARNO_2019",
                station_name="61",
            )

    def test_invalid_campaign_name(self, disdrodb_metadata_archive_dir):
        """Test error is raised with invalid campaign name."""
        with pytest.raises(ValueError):
            check_station_inputs(
                # DISDRODB root directories
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                # Station arguments
                data_source="EPFL",
                campaign_name="LOCARNO_2019_WRONG",
                station_name="61",
            )

    def test_invalid_station_name(self, disdrodb_metadata_archive_dir):
        """Test error is raised with invalid station name."""
        with pytest.raises(ValueError):
            check_station_inputs(
                # DISDRODB root directories
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                # Station arguments
                data_source="EPFL",
                campaign_name="LOCARNO_2019",
                station_name="60_WRONG",
            )


class TestHasAvailableData:
    """Tests for has_available_data function."""

    def test_directory_does_not_exist(self, tmp_path):
        """Test returns False if product directory does not exist."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")

        res = has_available_data(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="L0A",
            data_archive_dir=data_archive_dir,
        )
        assert res is False

    def test_directory_exists_but_empty(self, tmp_path):
        """Test returns False if directory exists but has no files."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")
        data_dir = define_data_dir(
            product="L0A",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=data_archive_dir,
            check_exists=False,
        )
        os.makedirs(data_dir, exist_ok=True)
        res = has_available_data(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="L0A",
            data_archive_dir=data_archive_dir,
        )
        assert res is False

    def test_directory_with_files(self, tmp_path):
        """Test returns True if directory contains files."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")
        data_dir = define_data_dir(
            product="L0A",
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            check_exists=False,
        )
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, "dummy.txt")
        with open(file_path, "w") as f:
            f.write("test")

        res = has_available_data(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="L0A",
            data_archive_dir=data_archive_dir,
        )
        assert res is True


class TestCheckDataAvailability:
    """Tests for has_available_data function."""

    def test_raise_error_if_directory_does_not_exist(self, tmp_path):
        """Test raise error if product directory does not exist."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")

        with pytest.raises(ValueError):
            check_data_availability(
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="L0A",
                data_archive_dir=data_archive_dir,
            )

    def test_raise_error_if_directory_exists_but_empty(self, tmp_path):
        """Test raise error if product directory exists but has no files."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")
        data_dir = define_data_dir(
            product="L0A",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=data_archive_dir,
            check_exists=False,
        )
        os.makedirs(data_dir, exist_ok=True)
        with pytest.raises(ValueError):
            check_data_availability(
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="L0A",
                data_archive_dir=data_archive_dir,
            )

    def test_valid_directory_with_files(self, tmp_path):
        """Test do not raise error if directory contains files."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")
        data_dir = define_data_dir(
            product="L0A",
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            check_exists=False,
        )
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, "dummy.txt")
        with open(file_path, "w") as f:
            f.write("test")

        check_data_availability(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="L0A",
            data_archive_dir=data_archive_dir,
        )


class TestCheckMetadata_file:
    """Test check_metadata_file function."""

    def test_valid_metadata(self, disdrodb_metadata_archive_dir):
        """Test no error is raised with valid station metadata."""
        metadata_filepath = check_metadata_file(
            # DISDRODB root directories
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            # Station arguments
            data_source="EPFL",
            campaign_name="LOCARNO_2019",
            station_name="61",
        )
        assert isinstance(metadata_filepath, str)
        assert metadata_filepath.endswith("61.yml")

    def test_missing_metadata(self, disdrodb_metadata_archive_dir):
        """Test raise error if missing station metadata."""
        with pytest.raises(ValueError):
            check_metadata_file(
                # DISDRODB root directories
                metadata_archive_dir=disdrodb_metadata_archive_dir,
                # Station arguments
                data_source="EPFL",
                campaign_name="LOCARNO_2019",
                station_name="NOT_EXIST",
            )


class TestCheckIssueDir:
    """Tests for check_issue_dir function."""

    def test_issue_dir_exists(self, tmp_path):
        """Test check_issue_dir returns existing issue directory."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Create expected directory structure
        issue_dir = define_issue_dir(
            data_source=data_source,
            campaign_name=campaign_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=False,
        )
        os.makedirs(issue_dir, exist_ok=True)

        result = check_issue_dir(
            data_source=data_source,
            campaign_name=campaign_name,
            metadata_archive_dir=metadata_archive_dir,
        )
        assert os.path.exists(result)
        assert result == issue_dir

    def test_raise_no_error_if_issue_dir_not_exists(self, tmp_path):
        """Test check_issue_dir still returns path if directory missing (no error)."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")
        issue_dir = define_issue_dir(
            data_source=data_source,
            campaign_name=campaign_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=False,
        )
        # Do not create the directory
        result = check_issue_dir(
            data_source=data_source,
            campaign_name=campaign_name,
            metadata_archive_dir=metadata_archive_dir,
        )
        assert not os.path.exists(result)
        assert result == issue_dir


class TestCheckIssueFile:
    """Tests for check_issue_file function."""

    def test_issue_file_created_and_valid(self, tmp_path):
        """Test check_issue_file creates missing file and validates it."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Create parent issue dir
        issue_dir = define_issue_dir(
            data_source=data_source,
            campaign_name=campaign_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=False,
        )
        os.makedirs(issue_dir, exist_ok=True)

        # File path expected
        issue_filepath = define_issue_filepath(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=False,
        )

        # Ensure file does not exist initially
        assert not os.path.exists(issue_filepath)

        # Function should create and validate it
        result = check_issue_file(
            data_source,
            campaign_name,
            station_name,
            metadata_archive_dir=metadata_archive_dir,
        )
        assert os.path.exists(result)
        assert result == issue_filepath

    def test_issue_file_already_exists(self, tmp_path):
        """Test check_issue_file returns existing issue file."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")

        issue_dir = define_issue_dir(
            data_source=data_source,
            campaign_name=campaign_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=False,
        )
        os.makedirs(issue_dir, exist_ok=True)

        issue_filepath = define_issue_filepath(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=False,
        )
        # Create an empty YAML file manually
        with open(issue_filepath, "w") as f:
            f.write("")

        result = check_issue_file(
            data_source,
            campaign_name,
            station_name,
            metadata_archive_dir=metadata_archive_dir,
        )
        assert result == issue_filepath
