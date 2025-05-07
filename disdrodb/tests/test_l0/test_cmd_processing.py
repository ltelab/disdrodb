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
"""Test DISDRODB L0 processing commands."""

import os
import shutil

import pytest
from click.testing import CliRunner

from disdrodb import __root_path__
from disdrodb.api.path import define_data_dir
from disdrodb.cli.disdrodb_run_l0 import disdrodb_run_l0
from disdrodb.cli.disdrodb_run_l0_station import disdrodb_run_l0_station
from disdrodb.cli.disdrodb_run_l0a import disdrodb_run_l0a
from disdrodb.cli.disdrodb_run_l0a_station import disdrodb_run_l0a_station
from disdrodb.cli.disdrodb_run_l0b import disdrodb_run_l0b
from disdrodb.cli.disdrodb_run_l0b_station import disdrodb_run_l0b_station
from disdrodb.routines import (
    run_disdrodb_l0_station,
    run_disdrodb_l0a,
    run_disdrodb_l0a_station,
    run_disdrodb_l0b,
    run_disdrodb_l0b_station,
)
from disdrodb.utils.directories import count_files

BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")
DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "PARSIVEL_2007"
STATION_NAME = "10"
DEBUGGING_MODE = True
VERBOSE = False
FORCE = False

# from disdrodb.metadata.download import download_metadata_archive
# import pathlib

# tmp_path = pathlib.Path("/tmp/17/")
# tmp_path.mkdir(parents=True)
# test_data_archive_dir = tmp_path / "data" / "DISDRODB"
# shutil.copytree(BASE_DIR, test_data_archive_dir)

# parallel = False
# test_metadata_archive_dir = download_metadata_archive(tmp_path / "original_metadata_archive_repo")


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l0a_station(tmp_path, disdrodb_metadata_archive_dir, parallel, cli):
    """Test the disdrodb_run_l0a_station command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive
    shutil.copytree(BASE_DIR, test_data_archive_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0a_station,
            [
                # Station arguments
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                # Processing options
                "--parallel",
                parallel,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--verbose",
                VERBOSE,
                "--force",
                FORCE,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        run_disdrodb_l0a_station(
            # Station arguments
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            # Processing options
            parallel=parallel,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.parquet", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l0b_station(tmp_path, disdrodb_metadata_archive_dir, parallel, cli):
    """Test the disdrodb_run_l0b_station command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    shutil.copytree(BASE_DIR, test_data_archive_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0a_station,
            [
                # Station arguments
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                # Processing options
                "--parallel",
                parallel,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--verbose",
                VERBOSE,
                "--force",
                FORCE,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
        runner.invoke(
            disdrodb_run_l0b_station,
            [  # Station arguments
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                # Processing options
                "--parallel",
                parallel,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        run_disdrodb_l0a_station(
            # Station arguments
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            # Processing options
            parallel=parallel,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )

        run_disdrodb_l0b_station(
            # Station arguments
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            # Processing options
            parallel=parallel,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_disdrodb_run_l0_nc_station(tmp_path, disdrodb_metadata_archive_dir, verbose, parallel, cli):
    """Test the disdrodb_run_l0_station process correctly raw netCDF files."""
    DATA_SOURCE = "UK"
    CAMPAIGN_NAME = "DIVEN"
    STATION_NAME = "CAIRNGORM"

    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive
    shutil.copytree(BASE_DIR, test_data_archive_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0_station,
            [
                # Station arguments
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                # Processing options
                "--verbose",
                verbose,
                "--parallel",
                parallel,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        run_disdrodb_l0_station(
            # Station arguments
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            # Processing options
            parallel=parallel,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_disdrodb_run_l0_station(tmp_path, disdrodb_metadata_archive_dir, verbose, cli):
    """Test the disdrodb_run_l0_station command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    shutil.copytree(BASE_DIR, test_data_archive_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0_station,
            [
                # Station arguments
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                # Processing options
                "--verbose",
                verbose,
                "--parallel",
                False,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        run_disdrodb_l0_station(
            # Station arguments
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=verbose,
            debugging_mode=DEBUGGING_MODE,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_run_l0a(tmp_path, disdrodb_metadata_archive_dir, cli):
    """Test the disdrodb_run_l0a command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    shutil.copytree(BASE_DIR, test_data_archive_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0a,
            [
                # Stations options
                "--data_sources",
                DATA_SOURCE,
                "--campaign_names",
                CAMPAIGN_NAME,
                "--station_names",
                STATION_NAME,
                # Processing options
                "--verbose",
                VERBOSE,
                "--parallel",
                False,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        run_disdrodb_l0a(
            # Station options
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.parquet", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_run_l0b(tmp_path, disdrodb_metadata_archive_dir, cli):
    """Test the disdrodb_run_l0b command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    shutil.copytree(BASE_DIR, test_data_archive_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0a,
            [
                # Stations options
                "--data_sources",
                DATA_SOURCE,
                "--campaign_names",
                CAMPAIGN_NAME,
                "--station_names",
                STATION_NAME,
                # Processing options
                "--verbose",
                VERBOSE,
                "--parallel",
                False,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )

        runner.invoke(
            disdrodb_run_l0b,
            [
                # Station options
                "--data_sources",
                DATA_SOURCE,
                "--campaign_names",
                CAMPAIGN_NAME,
                "--station_names",
                STATION_NAME,
                # Processing options
                "--verbose",
                VERBOSE,
                "--parallel",
                False,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        run_disdrodb_l0a(
            # Stations options
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )
        run_disdrodb_l0b(
            # Station options
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("remove_l0a", [True, False])
@pytest.mark.parametrize("remove_l0b", [True, False])
def test_disdrodb_run_l0(tmp_path, disdrodb_metadata_archive_dir, remove_l0a, remove_l0b):
    """Test the disdrodb_run_l0b command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    shutil.copytree(BASE_DIR, test_data_archive_dir)

    # Produce data
    runner = CliRunner()
    runner.invoke(
        disdrodb_run_l0,
        [
            # DISDRODB root directories
            "--data_archive_dir",
            test_data_archive_dir,
            "--metadata_archive_dir",
            test_metadata_archive_dir,
            # Processing options
            "--debugging_mode",
            True,
            "--remove_l0a",
            remove_l0a,
            "--remove_l0b",
            remove_l0b,
        ],
    )

    # Check files are produced
    l0a_data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    l0b_data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    l0c_data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L0C",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    if remove_l0a:
        assert count_files(l0a_data_dir, glob_pattern="*.parquet", recursive=True) == 0
    else:
        assert count_files(l0a_data_dir, glob_pattern="*.parquet", recursive=True) > 0

    if remove_l0b:
        assert count_files(l0b_data_dir, glob_pattern="*.nc", recursive=True) == 0
    else:
        assert count_files(l0b_data_dir, glob_pattern="*.nc", recursive=True) > 0

    assert count_files(l0c_data_dir, glob_pattern="*.nc", recursive=True) > 0
