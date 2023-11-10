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
"""Check readers."""

import glob
import os
import shutil

import pandas as pd

from disdrodb import __root_path__
from disdrodb.utils.directories import list_files
from disdrodb.api.io import get_disdrodb_path
from disdrodb.l0.l0_reader import get_station_reader_function

TEST_BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")


def _get_list_test_data_sources() -> list:
    """Get list of test data sources.

    Returns
    -------
    list
        List of test data sources.
    """

    list_of_data_sources = os.listdir(os.path.join(TEST_BASE_DIR, "Raw"))
    return list_of_data_sources


def _get_list_test_campaigns(data_source: str) -> list:
    """Get list of test campaigns for a given data source.

    Parameters
    ----------
    data_source : str
        Data source.

    Returns
    -------
    list
        List of test campaigns.

    """
    list_of_campaigns = os.listdir(os.path.join(TEST_BASE_DIR, "Raw", data_source))
    return list_of_campaigns


def _get_list_test_stations(data_source: str, campaign_name: str) -> list:
    """Get list of test stations for a given data source and campaign.

    Parameters
    ----------
    data_source : str
        Data source.

    campaign_name : str
        Name of the campaign.

    Returns
    -------
    list
        List of test stations.

    """
    metadata_dir = os.path.join(TEST_BASE_DIR, "Raw", data_source, campaign_name, "metadata")
    filepaths = list_files(metadata_dir, glob_pattern="*.yml", recursive=False)
    list_station_names = [os.path.splitext(os.path.basename(i))[0] for i in filepaths]

    return list_station_names


def _is_parquet_files_identical(file1: str, file2: str) -> bool:
    """Check if two parquet files are identical.

    Parameters
    ----------
    file1 : str
        Path to the first file.

    file2 : str
        Path to the second file.

    Returns
    -------
    bool
        True if the two files are identical, False otherwise.

    """
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)
    return df1.equals(df2)


def _run_reader_on_test_data(data_source: str, campaign_name: str) -> None:
    """Run reader over the test data sample.

    Parameters
    ----------
    data_source : str
        Data source.
    campaign_name : str
        Campaign name.
    """
    station_names = _get_list_test_stations(data_source=data_source, campaign_name=campaign_name)
    for station_name in station_names:
        reader = get_station_reader_function(
            base_dir=TEST_BASE_DIR,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

        # Define campaign_name raw_dir and process_dir
        raw_dir = get_disdrodb_path(
            base_dir=TEST_BASE_DIR,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
        )

        processed_dir = get_disdrodb_path(
            base_dir=TEST_BASE_DIR,
            product="L0A",
            data_source=data_source,
            campaign_name=campaign_name,
            check_exists=False,
        )
        # Call the reader
        reader(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            station_name=station_name,
            force=True,
            verbose=False,
            debugging_mode=False,
            parallel=False,
        )

        return processed_dir


def check_all_readers() -> None:
    """Test all readers that have data samples and ground truth.

    Raises
    ------
    Exception
        If the reader validation has failed.
    """

    for data_source in _get_list_test_data_sources():
        for campaign_name in _get_list_test_campaigns(data_source):
            process_dir = _run_reader_on_test_data(data_source, campaign_name)
            ground_truth_dir = os.path.join(TEST_BASE_DIR, "Raw", data_source, campaign_name, "ground_truth")
            processed_product_dir = os.path.join(process_dir, "L0A")

            glob_pattern = os.path.join("*", "*.parquet")
            ground_truth_files = list_files(ground_truth_dir, glob_pattern=glob_pattern, recursive=False)
            processed_files = list_files(processed_product_dir, glob_pattern=glob_pattern, recursive=False)
        
            for ground_truth_fpath, processed_file_fpath in zip(ground_truth_files, processed_files):
                station_name = os.path.basename(os.path.dirname(ground_truth_fpath))
                is_correct = _is_parquet_files_identical(ground_truth_fpath, processed_file_fpath)
                if not is_correct:
                    raise Exception(
                        f"Reader validation has failed for data_source '{data_source}', campaign_name '{campaign_name}'"
                        f" and station_name '{station_name}'"
                    )

    # Remove Processed directory if exists
    if os.path.exists(os.path.join(TEST_BASE_DIR, "Processed")):
        try:
            shutil.rmtree(os.path.join(TEST_BASE_DIR, "Processed"))
        except Exception:
            pass
