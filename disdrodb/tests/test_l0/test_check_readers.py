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
"""Check DISDRODB L0 readers."""
import os
import shutil

import pandas as pd
import xarray as xr

from disdrodb import __root_path__
from disdrodb.api.io import available_stations
from disdrodb.api.path import define_campaign_dir, define_station_dir
from disdrodb.l0.l0_processing import run_l0a_station
from disdrodb.metadata import read_station_metadata
from disdrodb.utils.directories import list_files


def _check_identical_netcdf_files(file1: str, file2: str) -> bool:
    """Check if two L0B netCDF files are identical.

    Parameters
    ----------
    file1 : str
        Path to the first file.

    file2 : str
        Path to the second file.
    """
    # Open files
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)
    # Remove attributes that depends on processing time
    ds1.attrs.pop("disdrodb_processing_date", None)
    ds2.attrs.pop("disdrodb_processing_date", None)
    # Assert equality
    xr.testing.assert_identical(ds1, ds2)


def _check_identical_parquet_files(file1: str, file2: str) -> bool:
    """Check if two parquet files are identical.

    Parameters
    ----------
    file1 : str
        Path to the first file.

    file2 : str
        Path to the second file.
    """
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)
    if not df1.equals(df2):
        raise ValueError("The two Parquet files differ.")


def _check_station_reader_results(
    base_dir,
    data_source,
    campaign_name,
    station_name,
):
    raw_dir = define_campaign_dir(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
    )

    run_l0a_station(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        force=True,
        verbose=False,
        debugging_mode=False,
        parallel=False,
    )

    metadata = read_station_metadata(
        base_dir=base_dir,
        product="L0A",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    raw_data_format = metadata["raw_data_format"]
    if raw_data_format == "netcdf":
        glob_pattern = "*.nc"
        check_identical_files = _check_identical_netcdf_files
        product = "L0B"
    else:  # raw_data_format == "txt"
        glob_pattern = "*.parquet"
        check_identical_files = _check_identical_parquet_files
        product = "L0A"

    ground_truth_station_dir = os.path.join(raw_dir, "ground_truth", station_name)
    processed_station_dir = define_station_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Retrieve files
    ground_truth_files = sorted(list_files(ground_truth_station_dir, glob_pattern=glob_pattern, recursive=True))
    processed_files = sorted(list_files(processed_station_dir, glob_pattern=glob_pattern, recursive=True))

    # Check same number of files
    n_groud_truth = len(ground_truth_files)
    n_processed = len(processed_files)
    if n_groud_truth != n_processed:
        raise ValueError(f"{n_groud_truth} ground truth files but only {n_processed} are prfoduced.")

    # Compare equality of files
    for ground_truth_filepath, processed_filepath in zip(ground_truth_files, processed_files):
        try:
            check_identical_files(ground_truth_filepath, processed_filepath)
        except Exception:
            raise ValueError(f"Reader validation has failed for '{data_source}' '{campaign_name}' '{station_name}'")


def test_check_all_readers(tmp_path) -> None:
    """Test all readers that have data samples and ground truth.

    Raises
    ------
    Exception
        If the reader validation has failed.
    """
    TEST_BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")

    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(TEST_BASE_DIR, test_base_dir)

    list_stations_info = available_stations(
        product="RAW",
        data_sources=None,
        campaign_names=None,
        return_tuple=True,
        base_dir=test_base_dir,
    )

    for data_source, campaign_name, station_name in list_stations_info:
        _check_station_reader_results(
            base_dir=test_base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
