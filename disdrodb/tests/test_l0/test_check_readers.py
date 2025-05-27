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
from disdrodb.api.path import define_campaign_dir, define_station_dir
from disdrodb.api.search import available_stations
from disdrodb.l0.routines import run_l0a_station
from disdrodb.metadata import read_station_metadata
from disdrodb.utils.directories import list_files

TEST_BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")


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
    ds1 = xr.open_dataset(file1, decode_timedelta=False)
    ds2 = xr.open_dataset(file2, decode_timedelta=False)

    # Remove attributes that depends on processing time
    attrs_varying = ["disdrodb_processing_date", "disdrodb_software_version"]
    attrs_modified_recently = []
    attr_to_remove = attrs_varying + attrs_modified_recently
    for key in attr_to_remove:
        ds1.attrs.pop(key, None)
        ds2.attrs.pop(key, None)

    # Assert equality without attributes
    xr.testing.assert_allclose(ds1, ds2)

    # Assert equality with attributes
    # xr.testing.assert_identical(ds1, ds2)


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
    data_archive_dir,
    metadata_archive_dir,
    data_source,
    campaign_name,
    station_name,
):
    raw_station_dir = define_campaign_dir(
        archive_dir=data_archive_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
    )

    run_l0a_station(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        force=True,
        verbose=True,
        debugging_mode=False,
        parallel=False,
    )

    metadata = read_station_metadata(
        metadata_archive_dir=metadata_archive_dir,
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

    ground_truth_station_dir = os.path.join(raw_station_dir, "ground_truth", station_name)
    processed_station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
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
        raise ValueError(f"{n_groud_truth} ground truth files but only {n_processed} are produced.")

    # Compare equality of files
    # ground_truth_filepath, processed_filepath = list( zip(ground_truth_files, processed_files))[0]
    for ground_truth_filepath, processed_filepath in zip(ground_truth_files, processed_files):
        try:
            check_identical_files(ground_truth_filepath, processed_filepath)
        except Exception as e:
            raise ValueError(
                f"Reader validation has failed for '{data_source}' '{campaign_name}' '{station_name}'. Error is: {e}",
            )


# from disdrodb.metadata.download import download_metadata_archive
# import pathlib

# tmp_path = pathlib.Path("/tmp/19/")
# tmp_path.mkdir(parents=True)
# test_data_archive_dir = tmp_path / "data" / "DISDRODB"
# shutil.copytree(TEST_BASE_DIR, test_data_archive_dir)

# parallel = False
# test_metadata_archive_dir = download_metadata_archive(tmp_path / "original_metadata_archive_repo")


def test_check_all_readers(tmp_path, disdrodb_metadata_archive_dir) -> None:
    """Test all readers that have data samples and ground truth.

    Raises
    ------
    Exception
        If the reader validation has failed.
    """
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive
    shutil.copytree(TEST_BASE_DIR, test_data_archive_dir)

    list_stations_info = available_stations(
        data_archive_dir=test_data_archive_dir,
        metadata_archive_dir=test_metadata_archive_dir,
        product="RAW",
        data_sources=None,
        campaign_names=None,
        return_tuple=True,
        available_data=True,
    )

    # data_source, campaign_name, station_name = list_stations_info[0]
    # data_source, campaign_name, station_name = list_stations_info[1]

    for data_source, campaign_name, station_name in list_stations_info:
        _check_station_reader_results(
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )


# def update_ground_truth_data():
#     """Update benchmark files."""
#     import pathlib

#     from disdrodb.metadata.download import download_metadata_archive

#     # Define test directories
#     tmp_path = pathlib.Path("/tmp/22/")
#     tmp_path.mkdir(parents=True)

#     data_archive_dir = tmp_path / "data" / "DISDRODB"
#     metadata_archive_dir = download_metadata_archive(tmp_path / "original_metadata_archive_repo")
#     shutil.copytree(TEST_BASE_DIR, data_archive_dir)

#     # List stations to test
#     list_stations_info = available_stations(
#         data_archive_dir=data_archive_dir,
#         metadata_archive_dir=metadata_archive_dir,
#         product="RAW",
#         data_sources=None,
#         campaign_names=None,
#         return_tuple=True,
#         available_data=True,
#     )

#     # data_source, campaign_name, station_name = list_stations_info[0]
#     for data_source, campaign_name, station_name in list_stations_info:
#         # Produce expected test file
#         run_l0a_station(
#             data_archive_dir=data_archive_dir,
#             metadata_archive_dir=metadata_archive_dir,
#             data_source=data_source,
#             campaign_name=campaign_name,
#             station_name=station_name,
#             force=True,
#             verbose=True,
#             debugging_mode=False,
#             parallel=False,
#         )

#         # Define L0 product which is produced
#         metadata = read_station_metadata(
#             metadata_archive_dir=metadata_archive_dir,
#             data_source=data_source,
#             campaign_name=campaign_name,
#             station_name=station_name,
#         )
#         raw_data_format = metadata["raw_data_format"]
#         if raw_data_format == "netcdf":
#             glob_pattern = "*.nc"
#             product = "L0B"
#         else:  # raw_data_format == "txt"
#             glob_pattern = "*.parquet"
#             product = "L0A"

#         # List produced test files
#         processed_station_dir = define_station_dir(
#             data_archive_dir=data_archive_dir,
#             product=product,
#             data_source=data_source,
#             campaign_name=campaign_name,
#             station_name=station_name,
#         )
#         processed_files = sorted(list_files(processed_station_dir, glob_pattern=glob_pattern, recursive=True))

#         # Define location of original ground_truth files in the repo
#         test_campaign_dir = define_campaign_dir(
#             archive_dir=TEST_BASE_DIR,
#             product="RAW",
#             data_source=data_source,
#             campaign_name=campaign_name,
#         )
#         ground_truth_station_dir = os.path.join(test_campaign_dir, "ground_truth", station_name)

#         # Remove old ground truth and recreate the directory
#         shutil.rmtree(ground_truth_station_dir)
#         os.makedirs(ground_truth_station_dir, exist_ok=True)

#         # Copy each of the processed files into the ground truth location
#         for src in processed_files:
#             dst = os.path.join(ground_truth_station_dir, os.path.basename(src))
#             shutil.copy2(src, dst)
