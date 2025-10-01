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
"""Test DISDRODB station summary commands."""

import os
import shutil

import pytest
from click.testing import CliRunner

from disdrodb import package_dir
from disdrodb.api.path import define_station_dir
from disdrodb.cli.disdrodb_create_summary import disdrodb_create_summary
from disdrodb.cli.disdrodb_create_summary_station import disdrodb_create_summary_station
from disdrodb.constants import ARCHIVE_VERSION
from disdrodb.routines import create_summary, create_summary_station
from disdrodb.utils.directories import count_files

TEST_DATA_L2E_DIR = os.path.join(package_dir, "tests", "data", "test_data_l2e")

DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "HYMEX_LTE_SOP2"
STATION_NAME = "10"
DEBUGGING_MODE = True
PARALLEL = False
VERBOSE = False
FORCE = False
TEMPORAL_RESOLUTION = "1MIN"

# import pathlib
# tmp_path = pathlib.Path("/tmp/13")
# test_data_archive_dir = os.path.join(tmp_path, "DISDRODB")
# dst_dir = os.path.join(test_data_archive_dir, ARCHIVE_VERSION)
# shutil.copytree(TEST_DATA_L2E_DIR, dst_dir, dirs_exist_ok=True)
# parallel=False
# VERBOSE=True
# os.environ["PYTEST_CURRENT_TEST"] = "1"


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_create_summary_station(tmp_path, cli):
    """Test the disdrodb_create_summary_station command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"

    dst_dir = test_data_archive_dir / ARCHIVE_VERSION
    shutil.copytree(TEST_DATA_L2E_DIR, dst_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_create_summary_station,
            [
                # Station arguments
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                # Processing options
                "--parallel",
                PARALLEL,
                "--temporal_resolution",
                TEMPORAL_RESOLUTION,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
            ],
        )
    else:
        create_summary_station(
            # Station arguments
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            # Processing options
            parallel=PARALLEL,
            temporal_resolution=TEMPORAL_RESOLUTION,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
        )

    # Check summary files are produced
    summary_dir = define_station_dir(
        data_archive_dir=test_data_archive_dir,
        product="SUMMARY",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )

    assert count_files(summary_dir, glob_pattern="Dataset.SpectrumStats*.nc", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Dataframe.L2E.*.parquet", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.DSD_Summary.*.csv", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.DSD_Summary.*.pdf", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.Events_Summary.*.csv", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.Events_Summary.*.pdf", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.Station_Summary.*.yaml", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Figure.*.png", recursive=True) > 4


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_create_summary(tmp_path, disdrodb_metadata_archive_dir, cli):
    """Test the disdrodb_run_l2e command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_data_archive_dir / ARCHIVE_VERSION
    shutil.copytree(TEST_DATA_L2E_DIR, dst_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_create_summary,
            [
                # Station arguments
                "--data_sources",
                DATA_SOURCE,
                "--campaign_names",
                CAMPAIGN_NAME,
                "--station_names",
                STATION_NAME,
                # Processing options
                "--parallel",
                PARALLEL,
                "--temporal_resolution",
                TEMPORAL_RESOLUTION,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        create_summary(
            # Stations arguments
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=PARALLEL,
            temporal_resolution=TEMPORAL_RESOLUTION,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )

    # Check summary files are produced
    summary_dir = define_station_dir(
        data_archive_dir=test_data_archive_dir,
        product="SUMMARY",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(summary_dir, glob_pattern="Dataset.SpectrumStats*.nc", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Dataframe.L2E.*.parquet", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.DSD_Summary.*.csv", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.DSD_Summary.*.pdf", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.Events_Summary.*.csv", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.Events_Summary.*.pdf", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Table.Station_Summary.*.yaml", recursive=True) == 1
    assert count_files(summary_dir, glob_pattern="Figure.*.png", recursive=True) > 4
