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
from disdrodb.l0.routines import (
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

# test_base_dir = "/tmp/new/DISDRODB"
# shutil.copytree(BASE_DIR, test_base_dir)
# parallel = False


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l0a_station(tmp_path, parallel, cli):
    """Test the disdrodb_run_l0a_station command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0a_station,
            [
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                "--base_dir",
                test_base_dir,
                "--parallel",
                parallel,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--verbose",
                VERBOSE,
                "--force",
                FORCE,
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
            base_dir=test_base_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.parquet", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l0b_station(tmp_path, parallel, cli):
    """Test the disdrodb_run_l0b_station command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0a_station,
            [
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                "--base_dir",
                test_base_dir,
                "--parallel",
                parallel,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--verbose",
                VERBOSE,
                "--force",
                FORCE,
            ],
        )
        runner.invoke(
            disdrodb_run_l0b_station,
            [
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                "--base_dir",
                test_base_dir,
                "--parallel",
                parallel,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
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
            base_dir=test_base_dir,
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
            base_dir=test_base_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_disdrodb_run_l0_nc_station(tmp_path, verbose, parallel, cli):
    """Test the disdrodb_run_l0_station process correctly raw netCDF files."""
    BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")
    DATA_SOURCE = "UK"
    CAMPAIGN_NAME = "DIVEN"
    STATION_NAME = "CAIRNGORM"

    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0_station,
            [
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                "--base_dir",
                test_base_dir,
                "--verbose",
                verbose,
                "--parallel",
                parallel,
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
            base_dir=test_base_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_disdrodb_run_l0_station(tmp_path, verbose, cli):
    """Test the disdrodb_run_l0_station command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0_station,
            [
                DATA_SOURCE,
                CAMPAIGN_NAME,
                STATION_NAME,
                "--base_dir",
                test_base_dir,
                "--verbose",
                verbose,
                "--parallel",
                False,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
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
            base_dir=test_base_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_run_l0a(tmp_path, cli):
    """Test the disdrodb_run_l0a command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0a,
            [
                "--base_dir",
                test_base_dir,
                "--data_sources",
                DATA_SOURCE,
                "--campaign_names",
                CAMPAIGN_NAME,
                "--station_names",
                STATION_NAME,
                "--verbose",
                VERBOSE,
                "--parallel",
                False,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
            ],
        )
    else:
        run_disdrodb_l0a(
            # Station arguments
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            base_dir=test_base_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.parquet", recursive=True) > 0


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_run_l0b(tmp_path, cli):
    """Test the disdrodb_run_l0b command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l0a,
            [
                "--base_dir",
                test_base_dir,
                "--data_sources",
                DATA_SOURCE,
                "--campaign_names",
                CAMPAIGN_NAME,
                "--station_names",
                STATION_NAME,
                "--verbose",
                VERBOSE,
                "--parallel",
                False,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
            ],
        )

        runner.invoke(
            disdrodb_run_l0b,
            [
                "--base_dir",
                test_base_dir,
                "--data_sources",
                DATA_SOURCE,
                "--campaign_names",
                CAMPAIGN_NAME,
                "--station_names",
                STATION_NAME,
                "--verbose",
                VERBOSE,
                "--parallel",
                False,
                "--debugging_mode",
                DEBUGGING_MODE,
                "--force",
                FORCE,
            ],
        )
    else:
        run_disdrodb_l0a(
            # Station arguments
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            base_dir=test_base_dir,
        )
        run_disdrodb_l0b(
            # Station arguments
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
            base_dir=test_base_dir,
        )

    # Check files are produced
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(data_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("remove_l0a", [True, False])
@pytest.mark.parametrize("remove_l0b", [True, False])
@pytest.mark.parametrize("l0b_concat", [True, False])
def test_disdrodb_run_l0(tmp_path, remove_l0a, remove_l0b, l0b_concat):
    """Test the disdrodb_run_l0b command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    # Produce data
    runner = CliRunner()
    runner.invoke(
        disdrodb_run_l0,
        [
            "--base_dir",
            test_base_dir,
            "--debugging_mode",
            True,
            "--remove_l0a",
            remove_l0a,
            "--remove_l0b",
            remove_l0b,
            "--l0b_concat",
            l0b_concat,
        ],
    )

    # Check files are produced
    l0a_data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    l0b_data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    if remove_l0a:
        assert count_files(l0a_data_dir, glob_pattern="*.parquet", recursive=True) == 0

    if not remove_l0a:
        assert count_files(l0a_data_dir, glob_pattern="*.parquet", recursive=True) > 0

    if l0b_concat:
        if remove_l0b:
            assert count_files(l0b_data_dir, glob_pattern="*.nc", recursive=True) == 0
        else:
            assert count_files(l0b_data_dir, glob_pattern="*.nc", recursive=True) > 0

    # If not L0B concat, do not remove L0B also if remove_l0b is specified !
    if not l0b_concat and remove_l0b:
        assert count_files(l0b_data_dir, glob_pattern="*.nc", recursive=True) > 0
