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
from disdrodb.api.path import define_station_dir
from disdrodb.l0.scripts.disdrodb_run_l0 import disdrodb_run_l0
from disdrodb.l0.scripts.disdrodb_run_l0_station import disdrodb_run_l0_station
from disdrodb.l0.scripts.disdrodb_run_l0a import disdrodb_run_l0a
from disdrodb.l0.scripts.disdrodb_run_l0a_station import disdrodb_run_l0a_station
from disdrodb.l0.scripts.disdrodb_run_l0b import disdrodb_run_l0b
from disdrodb.l0.scripts.disdrodb_run_l0b_station import disdrodb_run_l0b_station
from disdrodb.utils.directories import count_files

BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")
DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "PARSIVEL_2007"
STATION_NAME = "10"


@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l0a_station(tmp_path, parallel):
    """Test the disdrodb_run_l0a_station command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    runner = CliRunner()
    runner.invoke(
        disdrodb_run_l0a_station,
        [DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME, "--base_dir", str(test_base_dir), "--parallel", parallel],
    )

    station_dir = define_station_dir(
        base_dir=test_base_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(station_dir, glob_pattern="*.parquet", recursive=True) > 0


@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l0b_station(tmp_path, parallel):
    """Test the disdrodb_run_l0b_station command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    runner = CliRunner()
    runner.invoke(
        disdrodb_run_l0a_station,
        [DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME, "--base_dir", test_base_dir, "--parallel", parallel],
    )

    runner.invoke(
        disdrodb_run_l0b_station,
        [DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME, "--base_dir", test_base_dir, "--parallel", parallel],
    )

    station_dir = define_station_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(station_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("verbose", [True, False])
def test_disdrodb_run_l0_station(tmp_path, verbose):
    """Test the disdrodb_run_l0_station command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    runner = CliRunner()
    runner.invoke(
        disdrodb_run_l0_station,
        [DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME, "--base_dir", test_base_dir, "--verbose", verbose],
    )

    station_dir = define_station_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(station_dir, glob_pattern="*.nc", recursive=True) > 0


def test_disdrodb_run_l0a(tmp_path):
    """Test the disdrodb_run_l0a command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    runner = CliRunner()
    runner.invoke(disdrodb_run_l0a, ["--base_dir", test_base_dir])
    station_dir = define_station_dir(
        base_dir=test_base_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(station_dir, glob_pattern="*.parquet", recursive=True) > 0


def test_disdrodb_run_l0b(tmp_path):
    """Test the disdrodb_run_l0b command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    runner = CliRunner()
    runner.invoke(disdrodb_run_l0a, ["--base_dir", test_base_dir])

    runner.invoke(disdrodb_run_l0b, ["--base_dir", test_base_dir])

    station_dir = define_station_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(station_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("remove_l0a", [True, False])
@pytest.mark.parametrize("remove_l0b", [True, False])
@pytest.mark.parametrize("l0b_concat", [True, False])
def test_disdrodb_run_l0(tmp_path, remove_l0a, remove_l0b, l0b_concat):
    """Test the disdrodb_run_l0b command."""
    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

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

    l0a_station_dir = define_station_dir(
        base_dir=test_base_dir,
        product="L0A",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    l0b_station_dir = define_station_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    if remove_l0a:
        assert count_files(l0a_station_dir, glob_pattern="*.parquet", recursive=True) == 0

    if not remove_l0a:
        assert count_files(l0a_station_dir, glob_pattern="*.parquet", recursive=True) > 0

    if l0b_concat:
        if remove_l0b:
            assert count_files(l0b_station_dir, glob_pattern="*.nc", recursive=True) == 0
        else:
            assert count_files(l0b_station_dir, glob_pattern="*.nc", recursive=True) > 0

    # If not L0B concat, do not remove L0B also if remove_l0b is specified !
    if not l0b_concat:
        if remove_l0b:
            assert count_files(l0b_station_dir, glob_pattern="*.nc", recursive=True) > 0


@pytest.mark.parametrize("verbose", [True, False])
def test_disdrodb_run_l0_nc_station(tmp_path, verbose):
    """Test the disdrodb_run_l0_station process correctly raw netCDF files."""

    BASE_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "check_readers", "DISDRODB")
    DATA_SOURCE = "UK"
    CAMPAIGN_NAME = "DIVEN"
    STATION_NAME = "CAIRNGORM"

    test_base_dir = tmp_path / "DISDRODB"
    shutil.copytree(BASE_DIR, test_base_dir)

    runner = CliRunner()
    runner.invoke(
        disdrodb_run_l0_station,
        [DATA_SOURCE, CAMPAIGN_NAME, STATION_NAME, "--base_dir", test_base_dir, "--verbose", verbose],
    )

    station_dir = define_station_dir(
        base_dir=test_base_dir,
        product="L0B",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
    )
    assert count_files(station_dir, glob_pattern="*.nc", recursive=True) > 0
