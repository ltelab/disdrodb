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
"""Test DISDRODB Input/Output Function."""
import datetime
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import disdrodb
from disdrodb import __root_path__
from disdrodb.api import io
from disdrodb.api.io import (
    filter_by_time,
    find_files,
    open_data_archive,
    open_dataset,
    open_logs_directory,
    open_metadata_archive,
    open_metadata_directory,
    open_netcdf_files,
    open_product_directory,
    open_readers_directory,
    remove_product,
)
from disdrodb.api.path import define_data_dir
from disdrodb.constants import ARCHIVE_VERSION
from disdrodb.tests.conftest import create_fake_metadata_file, create_fake_raw_data_file

TEST_BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")
TEST_DATA_L0C_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "test_data_l0c")


# import pathlib
# tmp_path = pathlib.Path("/tmp/17")


class TestFindFiles:

    def test_find_raw_files(self, tmp_path):
        """Test finding raw files."""
        # Define station info
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"

        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        # Define correct glob pattern
        glob_pattern = "*.txt"

        # Test that the function raises an error if no files presenet
        with pytest.raises(ValueError, match="No RAW files are available in"):
            _ = find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="RAW",
                glob_pattern=glob_pattern,
            )

        # Add fake data files
        for filename in ["file1.txt", "file2.txt"]:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product="RAW",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )

        # Test that the function returns the correct number of files in debugging mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
            glob_pattern=glob_pattern,
            debugging_mode=True,
        )
        assert len(filepaths) == 2  # max(2, 3)

        # Add other fake data files
        for filename in ["file3.txt", "file4.txt"]:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product="RAW",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
            glob_pattern=glob_pattern,
            debugging_mode=True,
        )
        assert len(filepaths) == 3  # 3 when debugging_mode

        # Test that the function returns the correct number of files in normal mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
            glob_pattern=glob_pattern,
            debugging_mode=False,
        )
        assert len(filepaths) == 4

        # Test that the function raises an error if the glob_patterns is not a str or list
        with pytest.raises(ValueError, match="'glob_patterns' must be a str or list of strings."):
            find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="RAW",
                glob_pattern=1,
                debugging_mode=False,
            )

        # Test that the function raises an error if no files are found
        with pytest.raises(ValueError, match="No RAW files are available in"):
            find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="RAW",
                glob_pattern="*.csv",
                debugging_mode=False,
            )

        # Test with list of glob patterns
        for filename in ["file_new.csv"]:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product="RAW",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
            glob_pattern=["*txt", "*.csv"],
            debugging_mode=False,
        )
        assert len(filepaths) == 5

        # Test it get glob_pattern from metadata archive
        # - Create metadata file
        metadata_dict = {}
        metadata_dict["raw_data_glob_pattern"] = glob_pattern
        _ = create_fake_metadata_file(
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_dict=metadata_dict,
        )

        with disdrodb.config.set({"metadata_archive_dir": metadata_archive_dir}):
            filepaths = find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="RAW",
                # glob_pattern=glob_pattern, # took from metadata archive
                debugging_mode=False,
            )
            assert len(filepaths) == 4

    def test_find_l0a_files(self, tmp_path):
        """Test finding L0A files."""
        # Define station info
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0A"

        # Test that the function raises an error if no files presenet
        with pytest.raises(ValueError):
            _ = find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product=product,
            )

        # Add fake data files
        filenames = [
            "L0A.1MIN.LOCARNO_2019.61.s20190713134200.e20190714111000.V0.parquet",
            "L0A.1MIN.LOCARNO_2019.61.s20190714144200.e20190715111000.V0.parquet",
        ]
        for filename in filenames:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product=product,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )

        # Test that the function returns the correct number of files in debugging mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
            debugging_mode=True,
        )
        assert len(filepaths) == 2  # max(2, 3)

        # Test that the function returns the correct number of files in normal mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
        )
        assert len(filepaths) == 2

    def test_find_l0b_files(self, tmp_path):
        """Test finding L0B files."""
        # Define station info
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0B"

        # Test that the function raises an error if no files presenet
        with pytest.raises(ValueError):
            _ = find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product=product,
            )

        # Add fake data files
        filenames = [
            "L0B.1MIN.LOCARNO_2019.61.s20190713134200.e20190714111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190714144200.e20190715111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190715144200.e20190716111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190716144200.e20190717111000.V0.nc",
        ]
        for filename in filenames:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product=product,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )

        # Test that the function returns the correct number of files in debugging mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
            debugging_mode=True,
        )
        assert len(filepaths) == 3

        # Test that the function returns the correct number of files in normal mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
        )
        assert len(filepaths) == 4

        # Test it does not return other files except netCDFs
        for filename in [".hidden_file", "dummy.log"]:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product=product,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
        )
        assert len(filepaths) == 4

        # Test raise error if filtering by time filter out all files
        with pytest.raises(ValueError, match="No L0B files are available between 2022-01-01 and 2022-02-01"):
            find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product=product,
                start_time="2022-01-01",
                end_time="2022-02-01",
            )

        # Test raise error if there is a netCDF file with bad naming
        _ = create_fake_raw_data_file(
            data_archive_dir=data_archive_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            filename="dummy.nc",
        )
        with pytest.raises(ValueError, match="dummy.nc can not be parsed"):
            find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product=product,
            )


####--------------------------------------------------------------------------.
def test_filter_by_time() -> None:
    """Test filter filepaths."""
    filepaths = [
        "L2E.1MIN.LOCARNO_2019.61.s20190713134200.e20190731111000.V0.nc",
        "L2E.1MIN.LOCARNO_2019.61.s20190801001100.e20190831231600.V0.nc",
        "L2E.1MIN.LOCARNO_2019.61.s20200801001100.e20200831231600.V0.nc",
    ]

    out = filter_by_time(
        filepaths=filepaths,
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(out) == 2

    # Test None filepaths
    res = filter_by_time(
        filepaths=None,
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert res == []

    # Test empty filepath list
    res = filter_by_time(
        filepaths=[],
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert res == []

    # Test empty start time
    res = filter_by_time(
        filepaths=filepaths,
        start_time=None,
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(res) == 2

    # Test empty end time (should default to utcnow which will technically be
    # in the past by the time it gets to the function)
    res = filter_by_time(
        filepaths=filepaths,
        start_time=datetime.datetime(2019, 1, 1),
        end_time=None,
    )
    assert len(res) == 3

    # Test select when time period requested is within file
    res = filter_by_time(
        filepaths=filepaths,
        start_time=datetime.datetime(2019, 7, 14, 0, 0, 20),
        end_time=datetime.datetime(2019, 7, 15, 0, 0, 30),
    )
    assert len(res) == 1


class TestOpenDataset:
    """Test open_dataset."""

    def test_open_raw_txt_file(self, tmp_path, disdrodb_metadata_archive_dir):
        """Test using open_dataset to read raw text files."""
        from disdrodb.api.create_directories import create_l0_directory_structure
        from disdrodb.api.path import define_l0a_filename
        from disdrodb.l0.l0a_processing import write_l0a

        # Define DISDRODB root directories and station
        metadata_archive_dir = disdrodb_metadata_archive_dir
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        shutil.copytree(TEST_BASE_DIR, data_archive_dir)

        data_source = "EPFL"
        campaign_name = "PARSIVEL_2007"
        station_name = "10"

        # Open dataset
        df = open_dataset(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
        )

        # Test is a L0A pandas dataframe
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1

        # Write L0A
        data_dir = create_l0_directory_structure(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
            product="L0A",
            force=True,
        )
        filename = define_l0a_filename(df=df, campaign_name=campaign_name, station_name=station_name)
        filepath = os.path.join(data_dir, filename)
        write_l0a(df=df, filepath=filepath, force=True)

        # Test equivalent to open the L0A product with open_dataset
        df1 = open_dataset(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="L0A",
        )

        pd.testing.assert_frame_equal(df, df1)

    def test_open_raw_netcdf_file(self, tmp_path, disdrodb_metadata_archive_dir):
        """Test using open_dataset to read raw netCDF files."""
        # Define DISDRODB root directories and station
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        shutil.copytree(TEST_BASE_DIR, data_archive_dir)

        # Define station
        data_source = "UK"
        campaign_name = "DIVEN"
        station_name = "CAIRNGORM"

        # Open dataset
        ds = open_dataset(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
        )

        # Test is a L0B dataset
        assert isinstance(ds, xr.Dataset)

    def test_open_product_netcdf_files(self, tmp_path, disdrodb_metadata_archive_dir):
        """Test using open_dataset to read DISDRODB product netCDF files."""
        # Define station
        data_source = "EPFL"
        campaign_name = "HYMEX_LTE_SOP2"
        station_name = "10"

        # Define DISDRODB root directories and station
        data_archive_dir = tmp_path / "data" / "DISDRODB"

        # Prepare fake DISDRODB Data Archive
        dst_dir = data_archive_dir / ARCHIVE_VERSION
        shutil.copytree(TEST_DATA_L0C_DIR, dst_dir)

        # Open dataset
        ds = open_dataset(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="L0C",
        )

        # Test is a xarray dataset
        assert isinstance(ds, xr.Dataset)

    def test_open_product_filtered_netcdf_files(self, tmp_path, disdrodb_metadata_archive_dir):
        """Test using open_dataset to read DISDRODB product netCDF files."""
        # Define station
        data_source = "EPFL"
        campaign_name = "HYMEX_LTE_SOP2"
        station_name = "10"

        # Define DISDRODB root directories and station
        data_archive_dir = tmp_path / "data" / "DISDRODB"

        # Prepare fake DISDRODB Data Archive
        dst_dir = data_archive_dir / ARCHIVE_VERSION
        shutil.copytree(TEST_DATA_L0C_DIR, dst_dir)

        # Open dataset
        ds = open_dataset(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=disdrodb_metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            chunks=-1,
            variables="raw_drop_number",
            compute=True,
            product="L0C",
        )

        # Test is a xarray dataset
        assert isinstance(ds, xr.Dataset)


def test_open_netcdf_files_with_duplicate_timesteps(tmp_path):
    """Test open_netcdf_files deals correctly with duplicated timesteps."""
    # Create dataset with duplicated timesteps
    times = np.array(["2000-01-01 00:00:00", "2000-01-01 01:00:00", "2000-01-01 02:00:00"], dtype="M8[ns]")
    ds = xr.Dataset({"var": ("time", np.arange(len(times)))}, coords={"time": times})
    filepaths = [os.path.join(tmp_path, "dummy1.nc"), os.path.join(tmp_path, "dummy2.nc")]
    # Write two netcdf which are equal
    for filepath in filepaths:
        ds.to_netcdf(filepath)

    # Read with open_netcdf_files
    ds = open_netcdf_files(filepaths)
    # Test that duplicated timesteps are not dropped
    assert ds.sizes["time"] == 6, "open_netcdf_files drop duplicated timesteps!"
    # Test that filtering by times is allowed also in presence of duplicated timesteps
    ds = open_netcdf_files(filepaths, start_time="2000-01-01 00:00:00", end_time="2000-01-01 01:00:00")
    assert ds.sizes["time"] == 4


class TestRemoveProduct:
    """Unit tests for the remove_product function."""

    def test_raw_product_raises(self, tmp_path):
        """Test removal of RAW product raises ValueError."""
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        with pytest.raises(ValueError, match="Removal of 'RAW' files is not allowed."):
            remove_product(
                product="RAW",
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
            )

    def test_directory_is_removed(self, tmp_path):
        """Test product directory is removed from filesystem."""
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0A"
        # Define directory where product are saved
        data_dir = define_data_dir(
            data_archive_dir=data_archive_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

        # Create fake files
        for filename in ["file1.txt", "file2.txt"]:
            filepath = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product=product,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )

        assert os.path.exists(data_dir)
        assert os.path.exists(filepath)

        remove_product(
            data_archive_dir=data_archive_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

        assert not os.path.exists(filepath)
        assert not os.path.exists(data_dir)


class TestOpenDirectories:
    """Tests for DISDRODB open_* directory functions."""

    def test_open_logs_directory(self, tmp_path, monkeypatch):
        """Test logs directory path is passed to open_file_explorer."""
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"

        # Create fake logs dir
        logs_dir = data_archive_dir / ARCHIVE_VERSION / data_source / campaign_name / "logs"
        logs_dir.mkdir(parents=True)

        captured = {}
        monkeypatch.setattr(io, "open_file_explorer", lambda path: captured.setdefault("path", Path(path)))

        open_logs_directory(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=data_archive_dir,
        )

        assert captured["path"] == logs_dir

    def test_open_product_directory(self, tmp_path, monkeypatch):
        """Test product station directory path is passed to open_file_explorer."""
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0B"

        # Create fake station dir
        stat_dir = data_archive_dir / ARCHIVE_VERSION / data_source / campaign_name / product / station_name
        stat_dir.mkdir(parents=True)

        captured = {}
        monkeypatch.setattr(io, "open_file_explorer", lambda path: captured.setdefault("path", Path(path)))

        open_product_directory(
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=data_archive_dir,
        )

        assert captured["path"] == stat_dir

    def test_open_metadata_directory(self, tmp_path, monkeypatch):
        """Test metadata directory path is passed to open_file_explorer."""
        metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        meta_dir = metadata_archive_dir / "METADATA" / data_source / campaign_name / "metadata"
        meta_dir.mkdir(parents=True)

        captured = {}
        monkeypatch.setattr(io, "open_file_explorer", lambda path: captured.setdefault("path", Path(path)))

        open_metadata_directory(
            data_source=data_source,
            campaign_name=campaign_name,
            metadata_archive_dir=metadata_archive_dir,
        )

        assert captured["path"] == meta_dir

    def test_open_readers_directory(self, monkeypatch):
        """Test readers directory path is passed to open_file_explorer."""
        captured = {}
        monkeypatch.setattr(io, "open_file_explorer", lambda path: captured.setdefault("path", Path(path)))

        open_readers_directory()

        assert str(captured["path"]).endswith(os.path.join("disdrodb", "l0", "readers"))

    def test_open_metadata_archive(self, tmp_path, monkeypatch):
        """Test metadata archive root path is passed to open_file_explorer."""
        metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
        metadata_archive_dir.mkdir(parents=True)

        captured = {}
        monkeypatch.setattr(io, "open_file_explorer", lambda path: captured.setdefault("path", Path(path)))

        open_metadata_archive(metadata_archive_dir=metadata_archive_dir)

        assert captured["path"] == metadata_archive_dir

    def test_open_data_archive(self, tmp_path, monkeypatch):
        """Test data archive root path is passed to open_file_explorer."""
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_archive_dir.mkdir(parents=True)

        captured = {}
        monkeypatch.setattr(io, "open_file_explorer", lambda path: captured.setdefault("path", Path(path)))

        open_data_archive(data_archive_dir=data_archive_dir)

        assert captured["path"] == data_archive_dir
