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
"""Test DISDRODB info utility."""

import datetime
import os

import numpy as np
import pytest

from disdrodb.api.info import (
    check_groups,
    get_campaign_name_from_filepaths,
    get_end_time_from_filepaths,
    get_groups_value,
    get_info_from_filepath,
    get_key_from_filepath,
    get_key_from_filepaths,
    get_product_from_filepaths,
    get_sample_interval_from_filepaths,
    get_season,
    get_start_end_time_from_filepaths,
    get_start_time_from_filepaths,
    get_station_name_from_filepaths,
    get_time_component,
    get_version_from_filepaths,
    group_filepaths,
    infer_archive_dir_from_path,
    infer_campaign_name_from_path,
    infer_data_source_from_path,
    infer_disdrodb_tree_path,
    infer_disdrodb_tree_path_components,
    infer_path_info_dict,
    infer_path_info_tuple,
)
from disdrodb.api.path import define_filename

# Constants for testing
FILE_INFO = {
    "product": "L0A",
    "campaign_name": "LOCARNO_2018",
    "station_name": "60",
    "start_time": "20180625004331",
    "end_time": "20180711010000",
    "version": "1",
    "data_format": "parquet",
}

START_TIME = datetime.datetime.strptime(FILE_INFO["start_time"], "%Y%m%d%H%M%S")
END_TIME = datetime.datetime.strptime(FILE_INFO["end_time"], "%Y%m%d%H%M%S")
VALID_FNAME = (
    "{product:s}.{campaign_name:s}.{station_name:s}.s{start_time:s}.e{end_time:s}.{version:s}.{data_format:s}".format(
        **FILE_INFO,
    )
)
INVALID_FNAME = "invalid_filename.txt"
VALID_KEYS = ["product", "campaign_name", "station_name", "version", "data_format"]
INVALID_KEY = "nonexistent_key"

# valid_filepath = VALID_FNAME


@pytest.fixture
def valid_filepath(tmp_path):
    # Create a valid filepath for testing
    filepath = tmp_path / VALID_FNAME
    filepath.write_text("content does not matter")
    return str(filepath)


@pytest.fixture
def invalid_filepath(tmp_path):
    # Create an invalid filepath for testing
    filepath = tmp_path / INVALID_FNAME
    filepath.write_text("content does not matter")
    return str(filepath)


def test_infer_disdrodb_tree_path_components():
    """Test retrieve correct disdrodb path components."""
    archive_dir = os.path.join("whatever_path", "DISDRODB")
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    path_components = [archive_dir, "RAW", data_source, campaign_name]
    path = os.path.join(*path_components)
    assert infer_disdrodb_tree_path_components(path) == path_components

    with pytest.raises(ValueError):
        infer_disdrodb_tree_path_components("unvalid_path/because_not_disdrodb")


def test_infer_disdrodb_tree_path():
    # Assert retrieve correct disdrodb path
    disdrodb_path = os.path.join("DISDRODB", "RAW", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    assert infer_disdrodb_tree_path(path) == disdrodb_path

    # Assert raise error if not disdrodb path
    disdrodb_path = os.path.join("no_disdro_dir", "RAW", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    with pytest.raises(ValueError):
        infer_disdrodb_tree_path(path)

    # Assert raise error if not valid DISDRODB directory
    disdrodb_path = os.path.join("DISDRODB_UNVALID", "RAW", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_path", disdrodb_path)
    with pytest.raises(ValueError):
        infer_disdrodb_tree_path(path)

    # Assert it takes the right most DISDRODB occurrence
    disdrodb_path = os.path.join("DISDRODB", "RAW", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join("whatever_occurrence", "DISDRODB", "DISDRODB", "directory", disdrodb_path)
    assert infer_disdrodb_tree_path(path) == disdrodb_path

    # Assert behaviour when path == archive_dir
    archive_dir = os.path.join("home", "DISDRODB")
    assert infer_disdrodb_tree_path(archive_dir) == "DISDRODB"


def test_infer_archive_dir_from_path():
    # Assert retrieve correct disdrodb path
    archive_dir = os.path.join("whatever_path", "is", "before", "DISDRODB")
    disdrodb_path = os.path.join("RAW", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join(archive_dir, disdrodb_path)
    assert infer_archive_dir_from_path(path) == archive_dir

    # Assert raise error if not disdrodb path
    archive_dir = os.path.join("whatever_path", "is", "before", "NO_DISDRODB")
    disdrodb_path = os.path.join("RAW", "DATA_SOURCE", "CAMPAIGN_NAME")
    path = os.path.join(archive_dir, disdrodb_path)
    with pytest.raises(ValueError):
        infer_archive_dir_from_path(path)

    # Assert behaviour when path == archive_dir
    archive_dir = os.path.join("home", "DISDRODB")
    assert infer_archive_dir_from_path(archive_dir) == archive_dir


def test_infer_data_source_from_path():
    # Assert retrieve correct
    path = os.path.join("whatever_path", "DISDRODB", "RAW", "DATA_SOURCE", "CAMPAIGN_NAME")
    assert infer_data_source_from_path(path) == "DATA_SOURCE"

    # Assert raise error if path stop at RAW or ARCHIVE_VERSION
    path = os.path.join("whatever_path", "DISDRODB", "RAW")
    with pytest.raises(ValueError):
        infer_data_source_from_path(path)

    path = os.path.join("whatever_path", "DISDRODB", "ARCHIVE_VERSION")
    with pytest.raises(ValueError):
        infer_data_source_from_path(path)

    # Assert raise error if path not within DISDRODB
    path = os.path.join("whatever_path", "is", "not", "valid")
    with pytest.raises(ValueError):
        infer_data_source_from_path(path)


def test_infer_campaign_name_from_path():
    # Assert retrieve correct
    path = os.path.join("whatever_path", "DISDRODB", "RAW", "DATA_SOURCE", "CAMPAIGN_NAME", "...")
    assert infer_campaign_name_from_path(path) == "CAMPAIGN_NAME"

    # Assert raise error if path stop at RAW or ARCHIVE_VERSION
    path = os.path.join("whatever_path", "DISDRODB", "RAW")
    with pytest.raises(ValueError):
        infer_campaign_name_from_path(path)

    path = os.path.join("whatever_path", "DISDRODB", "ARCHIVE_VERSION")
    with pytest.raises(ValueError):
        infer_campaign_name_from_path(path)

    # Assert raise error if path not within DISDRODB
    path = os.path.join("whatever_path", "is", "not", "valid")
    with pytest.raises(ValueError):
        infer_campaign_name_from_path(path)


def test_infer_path_info_dict():
    # Assert retrieve correct
    archive_dir = os.path.join("whatever_path", "DISDRODB")
    path = os.path.join(archive_dir, "RAW", "DATA_SOURCE", "CAMPAIGN_NAME", "...")
    info_dict = infer_path_info_dict(path)
    assert info_dict["campaign_name"] == "CAMPAIGN_NAME"
    assert info_dict["data_source"] == "DATA_SOURCE"
    assert info_dict["data_archive_dir"] == archive_dir

    # Assert raise error if path stop at RAW or ARCHIVE_VERSION
    path = os.path.join("whatever_path", "DISDRODB", "RAW")
    with pytest.raises(ValueError):
        infer_path_info_dict(path)

    path = os.path.join("whatever_path", "DISDRODB", "ARCHIVE_VERSION")
    with pytest.raises(ValueError):
        infer_path_info_dict(path)


class TestInferPathInfoTuple:
    """Tests for infer_path_info_tuple."""

    def test_with_campaign_directory(self):
        """Test infer_path_info_tuple extracts archive, source and campaign correctly."""
        # Build realistic DISDRODB path
        data_archive_dir = os.path.join(os.sep, "dummy", "DISDRODB")
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        campaign_dir = os.path.join(data_archive_dir, "V0", data_source, campaign_name)

        # Call function
        result = infer_path_info_tuple(campaign_dir)

        # Validate
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result == (data_archive_dir, data_source, campaign_name)

    def test_with_filepath(self, tmp_path):
        """Test infer_path_info_tuple works when passing a file path inside archive."""
        # Build realistic DISDRODB filepath
        data_archive_dir = os.path.join(os.sep, "dummy", "DISDRODB")
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        file_path = os.path.join(
            data_archive_dir,
            "V0",
            data_source,
            campaign_name,
            "PRODUCT",
            "STATION_NAME",
            "file.nc",
        )

        result = infer_path_info_tuple(file_path)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result == (data_archive_dir, data_source, campaign_name)


def test_get_info_from_filepath(valid_filepath):
    # Test if the function correctly parses the file information
    info = get_info_from_filepath(valid_filepath)
    assert info["product"] == FILE_INFO["product"]
    assert info["campaign_name"] == FILE_INFO["campaign_name"]
    assert info["station_name"] == FILE_INFO["station_name"]


def test_get_info_from_filepath_raises_type_error(invalid_filepath):
    # Test if the function raises a TypeError for non-string input
    with pytest.raises(TypeError):
        get_info_from_filepath(1234)  # Intentional bad type


def test_get_info_from_filepath_raises_value_error(invalid_filepath):
    # Test if the function raises a ValueError for an unparsable filename
    with pytest.raises(ValueError):
        get_info_from_filepath(invalid_filepath)


@pytest.mark.parametrize("key", VALID_KEYS)
def test_get_key_from_filepath(valid_filepath, key):
    # Test if the function correctly retrieves the specific key
    product = get_key_from_filepath(valid_filepath, key)
    assert product == FILE_INFO[key]


def test_get_key_from_filepath_raises_key_error(valid_filepath):
    # Test if the function raises a KeyError for a non-existent key
    with pytest.raises(KeyError):
        get_key_from_filepath(valid_filepath, INVALID_KEY)


@pytest.mark.parametrize("key", VALID_KEYS)
def test_get_key_from_filepaths(valid_filepath, key):
    # Test if the function returns a list with the correct keys
    products = get_key_from_filepaths([valid_filepath, valid_filepath], key)
    assert products == [FILE_INFO[key], FILE_INFO[key]]


@pytest.mark.parametrize("key", VALID_KEYS)
def test_get_key_from_filepaths_single_path(valid_filepath, key):
    # Test if the function can handle a single filepath (string) as input
    products = get_key_from_filepaths(valid_filepath, key)
    assert products == [FILE_INFO[key]]


def test_get_version_from_filepath(valid_filepath):
    version = get_version_from_filepaths(valid_filepath)
    assert version == [FILE_INFO["version"]]


def test_get_version_from_filepath_raises_value_error(invalid_filepath):
    with pytest.raises(ValueError):
        get_version_from_filepaths(invalid_filepath)


def test_get_campaign_name_from_filepaths(valid_filepath):
    campaign_name = get_campaign_name_from_filepaths(valid_filepath)
    assert campaign_name == [FILE_INFO["campaign_name"]]


def test_get_station_name_from_filepaths(valid_filepath):
    station_name = get_station_name_from_filepaths(valid_filepath)
    assert station_name == [FILE_INFO["station_name"]]


def test_get_product_from_filepaths(valid_filepath):
    product = get_product_from_filepaths(valid_filepath)
    assert product == [FILE_INFO["product"]]


def test_get_start_time_from_filepaths(valid_filepath):
    start_time = get_start_time_from_filepaths(valid_filepath)
    assert start_time == [START_TIME]


def test_get_end_time_from_filepaths(valid_filepath):
    end_time = get_end_time_from_filepaths(valid_filepath)
    assert end_time == [END_TIME]


def test_get_start_end_time_from_filepaths(valid_filepath):
    start_time, end_time = get_start_end_time_from_filepaths(valid_filepath)
    assert np.array_equal(start_time, np.array([START_TIME]).astype("M8[s]"))
    assert np.array_equal(end_time, np.array([END_TIME]).astype("M8[s]"))


class TestGetSampleIntervalFromFilepaths:
    """Tests for get_sample_interval_from_filepaths."""

    def test_single_l2e_file(self):
        """Test returns list with correct interval for a single L2E file."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        end_time = datetime.datetime(2020, 1, 1, 12, 5, 0)
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        filepath = define_filename(
            product="L2E",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
            temporal_resolution="1MIN",
        )
        res = get_sample_interval_from_filepaths([filepath])
        assert res == [60]

    def test_multiple_l2e_files_same_interval(self):
        """Test returns list with same interval for multiple files."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        end_time = datetime.datetime(2020, 1, 1, 12, 5, 0)
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        filepaths = [
            define_filename(
                product="L2E",
                campaign_name=campaign_name,
                station_name=station_name,
                start_time=start_time,
                end_time=end_time,
                temporal_resolution="5MIN",
            ),
            define_filename(
                product="L2E",
                campaign_name=campaign_name,
                station_name=station_name,
                start_time=start_time,
                end_time=end_time,
                temporal_resolution="300S",
            ),
        ]
        res = get_sample_interval_from_filepaths(filepaths)
        assert res == [300, 300]

    def test_multiple_files_different_intervals(self):
        """Test returns list of different intervals when files differ."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        end_time = datetime.datetime(2020, 1, 1, 12, 5, 0)
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        file1 = define_filename(
            product="L2E",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
            temporal_resolution="1MIN",
        )
        file2 = define_filename(
            product="L2E",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
            temporal_resolution="5MIN",
        )
        res = get_sample_interval_from_filepaths([file1, file2])
        assert sorted(res) == [60, 300]


class TestCheckGroups:
    """Tests for check_groups."""

    def test_valid_string_and_list(self):
        """Test check_groups accepts valid string and list inputs."""
        assert check_groups("product") == ["product"]
        assert check_groups(["product", "year"]) == ["product", "year"]

    def test_invalid_type(self):
        """Test check_groups raises TypeError for invalid input type."""
        with pytest.raises(TypeError):
            check_groups(123)

    def test_invalid_key(self):
        """Test check_groups raises ValueError for invalid key."""
        with pytest.raises(ValueError):
            check_groups("invalid_key")


class TestGetSeason:
    """Tests for get_season."""

    def test_all_seasons(self):
        """Test get_season returns correct season by month."""
        assert get_season(datetime.date(2020, 1, 1)) == "DJF"
        assert get_season(datetime.date(2020, 4, 1)) == "MAM"
        assert get_season(datetime.date(2020, 7, 1)) == "JJA"
        assert get_season(datetime.date(2020, 10, 1)) == "SON"


class TestGetTimeComponent:
    """Tests for get_time_component."""

    def test_time_components(self):
        """Test get_time_component returns correct values for all keys."""
        t = datetime.datetime(2020, 4, 5, 15, 30, 45)  # April 5, 2020 Sunday
        assert get_time_component(t, "year") == "2020"
        assert get_time_component(t, "month") == "4"
        assert get_time_component(t, "day") == "5"
        assert get_time_component(t, "doy") == str(t.timetuple().tm_yday)
        assert get_time_component(t, "dow") == str(t.weekday())
        assert get_time_component(t, "hour") == "15"
        assert get_time_component(t, "minute") == "30"
        assert get_time_component(t, "second") == "45"
        assert get_time_component(t, "month_name") == "April"
        assert get_time_component(t, "quarter") == "2"
        assert get_time_component(t, "season") == "MAM"


class TestGetGroupsValue:
    """Tests for get_groups_value with real filenames."""

    def test_single_key(self):
        """Test returns string with e.g. start_time key."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        end_time = datetime.datetime(2020, 1, 1, 12, 5, 0)
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0A"

        filepath = define_filename(
            product=product,
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
        )
        res = get_groups_value(groups=["start_time"], filepath=filepath)
        assert isinstance(res, datetime.datetime)

    def test_multiple_keys(self):
        """Test returns combined string for multiple keys."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        end_time = datetime.datetime(2020, 1, 1, 12, 5, 0)
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L2E"

        filepath = define_filename(
            product=product,
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
            temporal_resolution="1MIN",
        )
        res = get_groups_value(["product", "year", "month"], filepath)
        assert res == "L2E/2020/1"


class TestGroupFilepaths:
    """Tests for group_filepaths with real filenames generated by define_filename."""

    def test_no_grouping_returns_input_list(self):
        """Test group_filepaths returns input list if groups=None."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        end_time = datetime.datetime(2020, 1, 1, 12, 5, 0)
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        filepath1 = define_filename(
            product="L0A",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
        )
        filepath2 = define_filename(
            product="L0A",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
        )

        filepaths = [filepath1, filepath2]
        assert group_filepaths(filepaths, groups=None) == filepaths

    def test_group_by_product(self):
        """Test group_filepaths groups filepaths by product key."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        end_time = datetime.datetime(2020, 1, 1, 12, 5, 0)
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        filepath1 = define_filename(
            product="L0A",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
        )
        filepath2 = define_filename(
            product="L2E",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
            temporal_resolution="1MIN",
        )
        filepath3 = define_filename(
            product="L2M",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
            temporal_resolution="1MIN",
            model_name="GAMMA",
        )

        filepaths = [filepath1, filepath2, filepath3]
        grouped = group_filepaths(filepaths, groups="product")
        assert "L0A" in grouped
        assert "L2E" in grouped
        assert "L2M" in grouped
        assert all(isinstance(v, list) for v in grouped.values())

    def test_group_by_year_and_season(self):
        """Test group_filepaths groups filepaths by year and season."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        end_time = datetime.datetime(2020, 1, 1, 12, 5, 0)
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        filepath1 = define_filename(
            product="L0A",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
        )
        filepath2 = define_filename(
            product="L0A",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
        )
        filepath3 = define_filename(
            product="L0A",
            campaign_name=campaign_name,
            station_name=station_name,
            start_time=start_time,
            end_time=end_time,
        )

        filepaths = [filepath1, filepath2, filepath3]
        grouped = group_filepaths(filepaths, groups=["year", "season"])
        keys = list(grouped.keys())
        assert any("2020" in k for k in keys)
        assert any(s in k for s in ["DJF", "MAM", "JJA", "SON"] for k in keys)
