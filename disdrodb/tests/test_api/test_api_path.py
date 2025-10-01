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
"""Test DISDRODB path."""
import datetime
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.api.path import (
    define_campaign_dir,
    define_config_dir,
    define_data_dir,
    define_data_source_dir,
    define_disdrodb_path,
    define_file_folder_path,
    define_filename,
    define_issue_dir,
    define_issue_filepath,
    define_l0a_filename,
    define_l0b_filename,
    define_l0c_filename,
    define_l1_filename,
    define_l2e_filename,
    define_l2m_filename,
    define_logs_dir,
    define_metadata_dir,
    define_metadata_filepath,
    define_partitioning_tree,
    define_product_dir_tree,
    define_station_dir,
    define_temporal_resolution,
)
from disdrodb.constants import ARCHIVE_VERSION


class TestDefineDisdrodbPath:
    """Tests for define_disdrodb_path and wrappers."""

    def test_metadata_path(self, tmp_path):
        """Test metadata path is built correctly."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        archive_dir = os.path.join(tmp_path, "DISDRODB")

        result = define_disdrodb_path(
            archive_dir,
            product="METADATA",
            data_source=data_source,
            campaign_name=campaign_name,
            check_exists=False,
        )
        expected = os.path.join(archive_dir, "METADATA", data_source, campaign_name)
        assert result == expected

    def test_raise_error_if_path_exists(self, tmp_path):
        """Test metadata path is built correctly."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Check do not raise error if do not exists if check_exists=False
        path = define_disdrodb_path(
            archive_dir,
            product="METADATA",
            data_source=data_source,
            campaign_name=campaign_name,
            check_exists=False,
        )

        # Check raise error if do not exists
        with pytest.raises(ValueError):
            define_disdrodb_path(
                archive_dir,
                product="METADATA",
                data_source=data_source,
                campaign_name=campaign_name,
                check_exists=True,
            )
        # Check do not raise error
        os.makedirs(path)
        define_disdrodb_path(
            archive_dir,
            product="METADATA",
            data_source=data_source,
            campaign_name=campaign_name,
            check_exists=True,
        )

    def test_raw_path(self, tmp_path):
        """Test raw path is built correctly."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        archive_dir = os.path.join(tmp_path, "DISDRODB")

        result = define_disdrodb_path(
            archive_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            check_exists=False,
        )
        expected = os.path.join(archive_dir, "RAW", data_source, campaign_name)
        assert result == expected

    def test_other_product_path(self, tmp_path):
        """Test other product path uses ARCHIVE_VERSION."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        archive_dir = os.path.join(tmp_path, "DISDRODB")

        result = define_disdrodb_path(
            archive_dir,
            product="L0A",
            data_source=data_source,
            campaign_name=campaign_name,
            check_exists=False,
        )
        expected = os.path.join(archive_dir, ARCHIVE_VERSION, data_source, campaign_name)
        assert result == expected

    def test_requires_datasource_with_campaign(self, tmp_path):
        """Test raises if campaign given without data_source."""
        campaign_name = "CAMPAIGN_NAME"

        archive_dir = os.path.join(tmp_path, "DISDRODB")
        with pytest.raises(ValueError):
            define_disdrodb_path(
                archive_dir,
                product="L0A",
                campaign_name=campaign_name,
                check_exists=False,
            )


class TestDefineDataSourceDir:
    """Test define_data_source_dir."""

    def test_define_data_source_dir(self, tmp_path):
        """Test define_data_source_dir returns correct path."""
        data_source = "DATA_SOURCE"

        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")

        result = define_data_source_dir(metadata_archive_dir, "METADATA", data_source)
        expected = os.path.join(metadata_archive_dir, "METADATA", data_source)
        assert result == expected

    def test_raise_error_if_data_source_dir_not_exists(self, tmp_path):
        """Test data source dir check_exists functionality."""
        data_source = "DATA_SOURCE"

        archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Check do not raise error if do not exists if check_exists=False
        path = define_data_source_dir(
            archive_dir,
            product="METADATA",
            data_source=data_source,
            check_exists=False,
        )

        # Check raise error if do not exists
        with pytest.raises(ValueError):
            define_data_source_dir(
                archive_dir,
                product="METADATA",
                data_source=data_source,
                check_exists=True,
            )

        # Check do not raise error
        os.makedirs(path)
        define_data_source_dir(
            archive_dir,
            product="METADATA",
            data_source=data_source,
            check_exists=True,
        )


class TestDefineCampaignDir:
    """Test define_campaign_dir."""

    def test_define_campaign_dir(self, tmp_path):
        """Test define_campaign_dir returns correct path."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")
        result = define_campaign_dir(metadata_archive_dir, "METADATA", data_source, campaign_name)
        expected = os.path.join(metadata_archive_dir, "METADATA", data_source, campaign_name)
        assert result == expected

    def test_raise_error_if_campaign_dir_not_exists(self, tmp_path):
        """Test campaign dir check_exists functionality."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Check do not raise error if do not exists if check_exists=False
        path = define_campaign_dir(
            archive_dir,
            product="METADATA",
            data_source=data_source,
            campaign_name=campaign_name,
            check_exists=False,
        )

        # Check raise error if do not exists
        with pytest.raises(ValueError):
            define_campaign_dir(
                archive_dir,
                product="METADATA",
                data_source=data_source,
                campaign_name=campaign_name,
                check_exists=True,
            )

        # Check do not raise error
        os.makedirs(path)
        define_campaign_dir(
            archive_dir,
            product="METADATA",
            data_source=data_source,
            campaign_name=campaign_name,
            check_exists=True,
        )


class TestDefineMetadataDir:
    """Test define_metadata_dir."""

    def test_define_metadata_dir(self, tmp_path):
        """Test metadata directory is defined correctly."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")

        result = define_metadata_dir(data_source, campaign_name, metadata_archive_dir=metadata_archive_dir)
        expected = os.path.join(metadata_archive_dir, "METADATA", data_source, campaign_name, "metadata")
        assert result == expected

    def test_raise_error_if_metadata_dir_not_exists(self, tmp_path):
        """Test metadata dir check_exists functionality."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        # Check do not raise error if do not exists if check_exists=False
        path = define_metadata_dir(
            data_source,
            campaign_name,
            metadata_archive_dir=os.path.join(tmp_path, "DISDRODB"),
            check_exists=False,
        )

        # Check raise error if do not exists
        with pytest.raises(ValueError):
            define_metadata_dir(
                data_source,
                campaign_name,
                metadata_archive_dir=os.path.join(tmp_path, "DISDRODB"),
                check_exists=True,
            )

        # Check do not raise error
        os.makedirs(path)
        define_metadata_dir(
            data_source,
            campaign_name,
            metadata_archive_dir=os.path.join(tmp_path, "DISDRODB"),
            check_exists=True,
        )


class TestDefineIssueDir:
    """Test define_issue_dir."""

    def test_define_issue_dir(self, tmp_path):
        """Test issue directory is defined correctly."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")

        result = define_issue_dir(data_source, campaign_name, metadata_archive_dir=metadata_archive_dir)
        expected = os.path.join(metadata_archive_dir, "METADATA", data_source, campaign_name, "issue")
        assert result == expected

    def test_raise_error_if_issue_dir_not_exists(self, tmp_path):
        """Test issue dir check_exists functionality."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        # Check do not raise error if do not exists if check_exists=False
        path = define_issue_dir(
            data_source,
            campaign_name,
            metadata_archive_dir=os.path.join(tmp_path, "DISDRODB"),
            check_exists=False,
        )

        # Check raise error if do not exists
        with pytest.raises(ValueError):
            define_issue_dir(
                data_source,
                campaign_name,
                metadata_archive_dir=os.path.join(tmp_path, "DISDRODB"),
                check_exists=True,
            )

        # Check do not raise error
        os.makedirs(path)
        define_issue_dir(
            data_source,
            campaign_name,
            metadata_archive_dir=os.path.join(tmp_path, "DISDRODB"),
            check_exists=True,
        )


class TestDefineConfigDir:
    """Tests define_config_dir."""

    def test_define_config_dir(self):
        """Test config dir is returned for L0 products."""
        path = define_config_dir("L0A")
        assert path.endswith(os.path.join("disdrodb", "l0", "configs"))

    def test_define_config_dir_not_implemented(self):
        """Test raises for unsupported product."""
        with pytest.raises(NotImplementedError):
            define_config_dir("L2E")


def test_define_partitioning_tree_year():
    """Test partitioning by year/month/day/quarter/name."""
    t = datetime.datetime(2020, 4, 15, 12, 0, 0)
    assert define_partitioning_tree(t, "") == ""
    assert define_partitioning_tree(t, "year") == "2020"
    assert define_partitioning_tree(t, "year/month") == os.path.join("2020", "04")
    assert define_partitioning_tree(t, "year/month/day") == os.path.join("2020", "04", "15")
    assert define_partitioning_tree(t, "year/month_name") == os.path.join("2020", "April")
    assert define_partitioning_tree(t, "year/quarter") == os.path.join("2020", "Q2")
    with pytest.raises(NotImplementedError):
        define_partitioning_tree(t, "bad")


class TestProductDirTree:
    """Tests for product_dir_tree."""

    def test_define_product_dir_tree_l0(self):
        """Test RAW and L0 product dir tree."""
        res = define_product_dir_tree("RAW")
        assert res == ""

        res = define_product_dir_tree("L0A")
        assert res == ""

        res = define_product_dir_tree("L0B")
        assert res == ""

        res = define_product_dir_tree("L0C")
        assert res == ""

    def test_define_product_dir_tree_l1(self):
        """Test L1 product dir tree."""
        res = define_product_dir_tree("L1", temporal_resolution="1MIN")
        assert res == "1MIN"

    def test_define_product_dir_tree_l2e(self):
        """Test L2E product dir tree."""
        res = define_product_dir_tree("L2E", temporal_resolution="1MIN")
        assert res == "1MIN"

    def test_define_product_dir_tree_l2m(self):
        """Test L2M product dir tree includes model_name."""
        res = define_product_dir_tree("L2M", temporal_resolution="1MIN", model_name="GAMMA")
        assert res == os.path.join("GAMMA", "1MIN")


class TestDefineLogsDir:
    """Test define_logs_dir."""

    def test_define_logs_dir(self, tmp_path):
        """Test logs dir path for L0A product."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        data_archive_dir = os.path.join(tmp_path, "DISDRODB")

        res = define_logs_dir("L0A", data_source, campaign_name, station_name, data_archive_dir=data_archive_dir)
        assert res.endswith(os.path.join("logs", "files", "L0A", station_name))

    def test_raise_error_if_path_exists_logs_dir(self, tmp_path):
        """Test logs dir check_exists functionality."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        data_archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Check do not raise error if do not exists if check_exists=False
        path = define_logs_dir(
            "L0A",
            data_source,
            campaign_name,
            station_name,
            data_archive_dir=data_archive_dir,
            check_exists=False,
        )

        # Check raise error if do not exists
        with pytest.raises(ValueError):
            define_logs_dir(
                "L0A",
                data_source,
                campaign_name,
                station_name,
                data_archive_dir=data_archive_dir,
                check_exists=True,
            )

        # Check do not raise error
        os.makedirs(path)
        define_logs_dir(
            "L0A",
            data_source,
            campaign_name,
            station_name,
            data_archive_dir=data_archive_dir,
            check_exists=True,
        )


class TestDefineDataDir:
    """Test define_data_dir path."""

    def test_define_data_dir(self, tmp_path):
        """Test data_dir path definition."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        data_archive_dir = os.path.join(tmp_path, "DISDRODB")

        data_dir = define_data_dir("L0A", data_source, campaign_name, station_name, data_archive_dir=data_archive_dir)
        assert data_dir.endswith(os.path.join("L0A", station_name))

    def test_raise_error_if_data_dir_not_exists(self, tmp_path):
        """Test data dir check_exists functionality."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        data_archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Check do not raise error if do not exists if check_exists=False
        path = define_data_dir(
            "L0A",
            data_source,
            campaign_name,
            station_name,
            data_archive_dir=data_archive_dir,
            check_exists=False,
        )

        # Check raise error if do not exists
        with pytest.raises(ValueError):
            define_data_dir(
                "L0A",
                data_source,
                campaign_name,
                station_name,
                data_archive_dir=data_archive_dir,
                check_exists=True,
            )

        # Check do not raise error
        os.makedirs(path)
        define_data_dir(
            "L0A",
            data_source,
            campaign_name,
            station_name,
            data_archive_dir=data_archive_dir,
            check_exists=True,
        )


class TestDefineStationDir:
    """Test define_station_dir path."""

    def test_define_station_dir(self, tmp_path):
        """Test define_station_dir path definition."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        data_archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Check for DISDRODB product
        station_dir = define_station_dir(
            "L0A",
            data_source,
            campaign_name,
            station_name,
            data_archive_dir=data_archive_dir,
        )
        assert station_dir.endswith(os.path.join("L0A", station_name))

        # Check for raw data
        station_dir = define_station_dir(
            "RAW",
            data_source,
            campaign_name,
            station_name,
            data_archive_dir=data_archive_dir,
        )
        assert station_dir.endswith(os.path.join("data", station_name))
        assert "RAW" in station_dir

    def test_raise_error_if_station_dir_not_exists(self, tmp_path):
        """Test station dir check_exists functionality."""
        data_archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Check do not raise error if do not exists if check_exists=False
        path = define_station_dir(
            "L0A",
            "DATA_SOURCE",
            "CAMPAIGN_NAME",
            "STATION_NAME",
            data_archive_dir=data_archive_dir,
            check_exists=False,
        )

        # Check raise error if do not exists
        with pytest.raises(ValueError):
            define_station_dir(
                "L0A",
                "DATA_SOURCE",
                "CAMPAIGN_NAME",
                "STATION_NAME",
                data_archive_dir=data_archive_dir,
                check_exists=True,
            )

        # Check do not raise error
        os.makedirs(path)
        define_station_dir(
            "L0A",
            "DATA_SOURCE",
            "CAMPAIGN_NAME",
            "STATION_NAME",
            data_archive_dir=data_archive_dir,
            check_exists=True,
        )


def test_define_temporal_resolution():
    """Test temporal resolution with and without rolling."""
    assert define_temporal_resolution(60, rolling=False) == "1MIN"
    assert define_temporal_resolution(60, rolling=True) == "ROLL1MIN"

    assert define_temporal_resolution(30, rolling=False) == "30S"
    assert define_temporal_resolution(30, rolling=True) == "ROLL30S"

    assert define_temporal_resolution(60 * 60, rolling=False) == "1H"
    assert define_temporal_resolution(60 * 60, rolling=True) == "ROLL1H"


class TestDefineFileFolderPath:
    """Tests for define_file_folder_path."""

    def test_with_dataframe(self, tmp_path):
        """Test works with pandas DataFrame input."""
        start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
        df = pd.DataFrame({"time": [start_time], "value": [1]})

        res = define_file_folder_path(df, dir_path=str(tmp_path), folder_partitioning="year/month/day")
        expected_suffix = os.path.join("2020", "01", "01")
        assert res.endswith(expected_suffix)

    def test_with_xarray_dataset(self, tmp_path):
        """Test works with xarray Dataset input."""
        start_time = datetime.datetime(2021, 6, 15, 15, 30, 0)
        ds = xr.Dataset(
            {"var": ("time", [1, 2])},
            coords={"time": [start_time, start_time + datetime.timedelta(minutes=5)]},
        )

        res = define_file_folder_path(ds, dir_path=str(tmp_path), folder_partitioning="year/month")
        expected_suffix = os.path.join("2021", "06")
        assert res.endswith(expected_suffix)

    def test_invalid_partitioning(self, tmp_path):
        """Test raises for invalid folder partitioning."""
        start_time = datetime.datetime(2022, 3, 1, 0, 0, 0)
        df = pd.DataFrame({"time": [start_time]})

        with pytest.raises(ValueError):
            define_file_folder_path(df, dir_path=str(tmp_path), folder_partitioning="bad_partitioning")


class TestDefineMetadataFilepath:
    """Tests for define_metadata_filepath."""

    def test_returns_correct_path_without_check(self, tmp_path):
        """Return correct metadata filepath when check_exists=False."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")
        filepath = define_metadata_filepath(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=False,
        )
        expected = os.path.join(
            metadata_archive_dir,
            "METADATA",
            data_source,
            campaign_name,
            "metadata",
            f"{station_name}.yml",
        )
        assert filepath == str(expected)

    def test_raises_if_file_missing_with_check(self, tmp_path):
        """Raise ValueError if metadata file is missing and check_exists=True."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")

        # Check raise error if missing
        with pytest.raises(ValueError):
            define_metadata_filepath(
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                metadata_archive_dir=str(metadata_archive_dir),
                check_exists=True,
            )
        # Check do not raise error if existing
        metadata_dir = os.path.join(metadata_archive_dir, "METADATA", data_source, campaign_name, "metadata")
        os.makedirs(metadata_dir)

        metadata_filepath = os.path.join(metadata_dir, f"{station_name}.yml")
        with open(metadata_filepath, "w") as f:
            f.write("dummy metadata")

        res = define_metadata_filepath(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=True,
        )
        assert res == str(metadata_filepath)


class TestDefineIssueFilepath:
    """Tests for define_issue_filepath."""

    def test_returns_correct_path_without_check(self, tmp_path):
        """Return correct issue filepath when check_exists=False."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")
        filepath = define_issue_filepath(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=False,
        )
        expected = os.path.join(
            metadata_archive_dir,
            "METADATA",
            data_source,
            campaign_name,
            "issue",
            f"{station_name}.yml",
        )
        assert filepath == expected

    def test_raises_if_file_missing_with_check(self, tmp_path):
        """Raise ValueError if issue file is missing and check_exists=True."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        metadata_archive_dir = os.path.join(tmp_path, "metadata", "DISDRODB")
        with pytest.raises(ValueError):
            define_issue_filepath(
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                metadata_archive_dir=metadata_archive_dir,
                check_exists=True,
            )

    def test_returns_path_if_file_exists_with_check(self, tmp_path):
        """Return issue filepath if file exists and check_exists=True."""
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        metadata_archive_dir = os.path.join(tmp_path, "DISDRODB")
        issue_dir = os.path.join(metadata_archive_dir, "METADATA", data_source, campaign_name, "issue")
        os.makedirs(issue_dir, exist_ok=True)
        issue_filepath = os.path.join(issue_dir, f"{station_name}.yml")
        with open(issue_filepath, "w") as f:
            f.write("dummy issue")

        res = define_issue_filepath(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
            check_exists=True,
        )
        assert res == issue_filepath


class TestDefineFilename:
    """Unit tests for the define_filename function."""

    def test_l0b_with_time_period(self):
        """Test filename generation for L0B with time period and default options."""
        start = datetime.datetime(2020, 1, 1, 0, 0, 0)
        end = datetime.datetime(2020, 1, 1, 1, 0, 0)
        fn = define_filename(
            product="L0B",
            campaign_name="CAMPAIGN_NAME",
            station_name="STATION_NAME",
            start_time=start,
            end_time=end,
        )
        assert fn == f"L0B.CAMPAIGN_NAME.STATION_NAME.s20200101000000.e20200101010000.{ARCHIVE_VERSION}.nc"

    def test_l0a_extension_parquet(self):
        """Test L0A product uses parquet extension instead of nc."""
        start = datetime.datetime(2021, 5, 1, 12, 0, 0)
        end = datetime.datetime(2021, 5, 1, 13, 0, 0)
        fn = define_filename(
            product="L0A",
            campaign_name="CAMPAIGN_NAME",
            station_name="STATION_NAME",
            start_time=start,
            end_time=end,
        )
        assert fn == f"L0A.CAMPAIGN_NAME.STATION_NAME.s20210501120000.e20210501130000.{ARCHIVE_VERSION}.parquet"

    def test_l0c_with_sample_interval(self):
        """Test L0C filename does not include temporal resolution."""
        start = datetime.datetime(2022, 6, 1, 0, 0, 0)
        end = datetime.datetime(2022, 6, 1, 0, 10, 0)
        fn = define_filename(
            product="L0C",
            campaign_name="CAMPAIGN_NAME",
            station_name="STATION_NAME",
            start_time=start,
            end_time=end,
        )
        assert fn == f"L0C.CAMPAIGN_NAME.STATION_NAME.s20220601000000.e20220601001000.{ARCHIVE_VERSION}.nc"

    def test_l2e_with_rolling(self):
        """Test L2E filename includes rolling temporal resolution."""
        start = datetime.datetime(2022, 7, 1, 0, 0, 0)
        end = datetime.datetime(2022, 7, 1, 0, 30, 0)
        fn = define_filename(
            product="L2E",
            campaign_name="CAMPAIGN_NAME",
            station_name="STATION_NAME",
            start_time=start,
            end_time=end,
            temporal_resolution="ROLL5MIN",
        )
        assert fn == f"L2E.ROLL5MIN.CAMPAIGN_NAME.STATION_NAME.s20220701000000.e20220701003000.{ARCHIVE_VERSION}.nc"

    def test_l2m_with_model_name(self):
        """Test L2M filename includes model name and temporal resolution."""
        start = datetime.datetime(2022, 8, 1, 0, 0, 0)
        end = datetime.datetime(2022, 8, 1, 1, 0, 0)
        fn = define_filename(
            product="L2M",
            campaign_name="CAMPAIGN_NAME",
            station_name="STATION_NAME",
            start_time=start,
            end_time=end,
            temporal_resolution="10MIN",
            model_name="GAMMA_ML",
        )
        assert (
            fn == f"L2M_GAMMA_ML.10MIN.CAMPAIGN_NAME.STATION_NAME.s20220801000000.e20220801010000.{ARCHIVE_VERSION}.nc"
        )

    def test_prefix_and_suffix(self):
        """Test prefix and suffix are added correctly."""
        start = datetime.datetime(2020, 9, 1, 0, 0, 0)
        end = datetime.datetime(2020, 9, 1, 1, 0, 0)
        fn = define_filename(
            product="L0B",
            campaign_name="CAMPAIGN_NAME",
            station_name="STATION_NAME",
            start_time=start,
            end_time=end,
            prefix="PRE",
            suffix="SUF",
        )
        assert fn == f"PRE.L0B.CAMPAIGN_NAME.STATION_NAME.s20200901000000.e20200901010000.{ARCHIVE_VERSION}.nc.SUF"

    def test_without_version_and_extension(self):
        """Test filename without version and extension."""
        start = datetime.datetime(2021, 1, 1, 0, 0, 0)
        end = datetime.datetime(2021, 1, 1, 0, 30, 0)
        fn = define_filename(
            product="L0B",
            campaign_name="CAMPAIGN_NAME",
            station_name="STATION_NAME",
            start_time=start,
            end_time=end,
            add_version=False,
            add_extension=False,
        )
        assert fn == "L0B.CAMPAIGN_NAME.STATION_NAME.s20210101000000.e20210101003000"

    def test_missing_time_raises(self):
        """Test ValueError is raised if add_time_period=True but times are missing."""
        with pytest.raises(ValueError):
            define_filename("L0B", "CAMPAIGN_NAME", "STATION_NAME")


def test_define_l0a_filename():
    """Test L0A filename generation."""
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    # Set variables
    product = "L0A"
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)

    # Create dataframe
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})

    # Define expected results
    expected_name = (
        f"{product}.{campaign_name}.{station_name}.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.parquet"
    )

    # Test the function
    res = define_l0a_filename(df, campaign_name, station_name)
    assert res == expected_name


def test_define_l0b_filename():
    """Test L0B filename generation."""
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    # Set variables
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)

    # Create xarray dataset
    timesteps = pd.date_range(start=start_date, end=end_date)
    data = np.zeros(timesteps.shape)
    ds = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": pd.date_range(start=start_date, end=end_date)},
    ).to_dataset(name="dummy")

    # Test the function
    fn = define_l0b_filename(ds, campaign_name, station_name)
    assert fn == f"L0B.CAMPAIGN_NAME.STATION_NAME.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.nc"


def test_define_l0c_filename():
    """Test L0C filename generation."""
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    # Set variables
    sample_interval = 60
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)

    # Create xarray dataset
    timesteps = pd.date_range(start=start_date, end=end_date)
    data = np.zeros(timesteps.shape)
    ds = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": pd.date_range(start=start_date, end=end_date), "sample_interval": sample_interval},
    ).to_dataset(name="dummy")

    # Test the function
    fn = define_l0c_filename(
        ds=ds,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    assert fn == f"L0C.1MIN.CAMPAIGN_NAME.STATION_NAME.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.nc"


def test_define_l1_filename():
    """Test L1 filename generation."""
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    # Set variables
    sample_interval = 60
    temporal_resolution = define_temporal_resolution(sample_interval, rolling=False)

    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)

    # Create xarray dataset
    timesteps = pd.date_range(start=start_date, end=end_date)
    data = np.zeros(timesteps.shape)
    ds = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": pd.date_range(start=start_date, end=end_date), "sample_interval": sample_interval},
    ).to_dataset(name="dummy")

    # Test the function
    fn = define_l1_filename(
        ds=ds,
        campaign_name=campaign_name,
        station_name=station_name,
        temporal_resolution=temporal_resolution,
    )
    assert fn == f"L1.1MIN.CAMPAIGN_NAME.STATION_NAME.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.nc"


def test_define_l2e_filename():
    """Test L2E filename generation."""
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    # Set variables
    sample_interval = 60
    rolling = True
    temporal_resolution = define_temporal_resolution(sample_interval, rolling=rolling)
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)

    # Create xarray dataset
    timesteps = pd.date_range(start=start_date, end=end_date)
    data = np.zeros(timesteps.shape)
    ds = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": pd.date_range(start=start_date, end=end_date), "sample_interval": sample_interval},
    ).to_dataset(name="dummy")

    # Test the function
    fn = define_l2e_filename(
        ds=ds,
        campaign_name=campaign_name,
        station_name=station_name,
        temporal_resolution=temporal_resolution,
    )
    assert fn == f"L2E.ROLL1MIN.CAMPAIGN_NAME.STATION_NAME.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.nc"


def test_define_l2m_filename():
    """Test L2M filename generation."""
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"

    # Set variables
    sample_interval = 60
    rolling = True
    temporal_resolution = define_temporal_resolution(sample_interval, rolling=rolling)

    model_name = "GAMMA_ML"
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)

    # Create xarray dataset
    timesteps = pd.date_range(start=start_date, end=end_date)
    data = np.zeros(timesteps.shape)
    ds = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": pd.date_range(start=start_date, end=end_date), "sample_interval": sample_interval},
    ).to_dataset(name="dummy")

    # Test the function
    fn = define_l2m_filename(
        ds=ds,
        campaign_name=campaign_name,
        station_name=station_name,
        temporal_resolution=temporal_resolution,
        model_name=model_name,
    )
    assert (
        fn == f"L2M_GAMMA_ML.ROLL1MIN.CAMPAIGN_NAME.STATION_NAME.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.nc"
    )
