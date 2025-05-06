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
"""Test DISDRODB L2 processing commands."""

import os
import shutil

import pytest
from click.testing import CliRunner

from disdrodb import ARCHIVE_VERSION, __root_path__
from disdrodb.api.path import define_data_dir
from disdrodb.cli.disdrodb_run_l2e import disdrodb_run_l2e
from disdrodb.cli.disdrodb_run_l2e_station import disdrodb_run_l2e_station
from disdrodb.cli.disdrodb_run_l2m import disdrodb_run_l2m
from disdrodb.cli.disdrodb_run_l2m_station import disdrodb_run_l2m_station
from disdrodb.routines import (
    run_disdrodb_l2e,
    run_disdrodb_l2e_station,
    run_disdrodb_l2m,
    run_disdrodb_l2m_station,
)
from disdrodb.utils.directories import count_files

TEST_DATA_L1_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "test_data_l1")
TEST_DATA_L2E_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "test_data_l2e")

DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "HYMEX_LTE_SOP2"
STATION_NAME = "10"
DEBUGGING_MODE = True
VERBOSE = False
FORCE = False


# from disdrodb.metadata.download import download_metadata_archive
# import pathlib
# tmp_path = pathlib.Path("/tmp/11")
# test_base_dir = os.path.join(tmp_path, "DISDRODB")
# dst_dir = os.path.join(test_base_dir, VERSION)
# shutil.copytree(TEST_DATA_L1_DIR, dst_dir, dirs_exist_ok=True)
# shutil.copytree(TEST_DATA_L2E_DIR, dst_dir, dirs_exist_ok=True)
# test_metadata_dir = download_metadata_archive(tmp_path / "original_metadata_archive_repo")


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l2e_station(tmp_path, disdrodb_metadata_dir, parallel, cli):
    """Test the disdrodb_run_l2e_station command."""
    test_base_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_dir = disdrodb_metadata_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_base_dir / ARCHIVE_VERSION
    shutil.copytree(TEST_DATA_L1_DIR, dst_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l2e_station,
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
                "--base_dir",
                test_base_dir,
                "--metadata_dir",
                test_metadata_dir,
            ],
        )
    else:
        run_disdrodb_l2e_station(
            # DISDRODB root directories
            base_dir=test_base_dir,
            metadata_dir=test_metadata_dir,
            # Station arguments
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            # Processing options
            parallel=parallel,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
        )

    # Check products at different sampling intervals are produced
    # - 1MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2E",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60,
        rolling=False,
    )
    assert count_files(data_dir, glob_pattern="L2E.1MIN.*.nc", recursive=True) == 2
    # - 10MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2E",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60 * 10,
        rolling=False,
    )
    assert count_files(data_dir, glob_pattern="L2E.10MIN.*.nc", recursive=True) == 2
    # - ROLL1MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2E",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60,
        rolling=True,
    )
    assert count_files(data_dir, glob_pattern="L2E.ROLL1MIN.*.nc", recursive=True) == 2


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l2m_station(tmp_path, disdrodb_metadata_dir, parallel, cli):
    """Test the disdrodb_run_l2m_station command."""
    test_base_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_dir = disdrodb_metadata_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_base_dir / ARCHIVE_VERSION
    shutil.copytree(TEST_DATA_L2E_DIR, dst_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l2m_station,
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
                "--base_dir",
                test_base_dir,
                "--metadata_dir",
                test_metadata_dir,
            ],
        )
    else:
        run_disdrodb_l2m_station(
            # DISDRODB root directories
            base_dir=test_base_dir,
            metadata_dir=test_metadata_dir,
            # Station arguments
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            # Processing options
            parallel=parallel,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
        )

    # Check products at different sampling intervals are produced
    # - 1MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2M",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60,
        rolling=False,
        model_name="GAMMA_ML",
    )
    assert count_files(data_dir, glob_pattern="L2M_GAMMA_ML.1MIN.*.nc", recursive=True) == 2
    # - ROLL1MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2M",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60,
        rolling=True,
        model_name="GAMMA_ML",
    )
    assert count_files(data_dir, glob_pattern="L2M_GAMMA_ML.ROLL1MIN.*.nc", recursive=True) == 2


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_run_l2e(tmp_path, disdrodb_metadata_dir, cli):
    """Test the disdrodb_run_l2e command."""
    test_base_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_dir = disdrodb_metadata_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_base_dir / ARCHIVE_VERSION
    shutil.copytree(TEST_DATA_L1_DIR, dst_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l2e,
            [
                # DISDRODB root directories
                "--base_dir",
                test_base_dir,
                "--metadata_dir",
                test_metadata_dir,
                # Station arguments
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
            ],
        )
    else:
        run_disdrodb_l2e(
            # DISDRODB root directories
            base_dir=test_base_dir,
            metadata_dir=test_metadata_dir,
            # Stations arguments
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
        )

    # Check products at different sampling intervals are produced
    # - 1MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2E",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60,
        rolling=False,
    )
    assert count_files(data_dir, glob_pattern="L2E.1MIN.*.nc", recursive=True) == 2
    # - 10MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2E",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60 * 10,
        rolling=False,
    )
    assert count_files(data_dir, glob_pattern="L2E.10MIN.*.nc", recursive=True) == 2
    # - ROLL1MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2E",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60,
        rolling=True,
    )
    assert count_files(data_dir, glob_pattern="L2E.ROLL1MIN.*.nc", recursive=True) == 2


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_run_l2m(tmp_path, disdrodb_metadata_dir, cli):
    """Test the disdrodb_run_l2m command."""
    test_base_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_dir = disdrodb_metadata_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_base_dir / ARCHIVE_VERSION
    shutil.copytree(TEST_DATA_L2E_DIR, dst_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l2m,
            [
                # DISDRODB root directories
                "--base_dir",
                test_base_dir,
                "--metadata_dir",
                test_metadata_dir,
                # Stations arguments
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
            ],
        )
    else:
        run_disdrodb_l2m(
            # DISDRODB root directories
            base_dir=test_base_dir,
            metadata_dir=test_metadata_dir,
            # Station arguments
            data_sources=DATA_SOURCE,
            campaign_names=CAMPAIGN_NAME,
            station_names=STATION_NAME,
            # Processing options
            parallel=False,
            force=FORCE,
            verbose=VERBOSE,
            debugging_mode=DEBUGGING_MODE,
        )

    # Check products at different sampling intervals are produced
    # - 1MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2M",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60,
        rolling=False,
        model_name="GAMMA_ML",
    )
    assert count_files(data_dir, glob_pattern="L2M_GAMMA_ML.1MIN.*.nc", recursive=True) == 2
    # - ROLL1MIN
    data_dir = define_data_dir(
        base_dir=test_base_dir,
        product="L2M",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        sample_interval=60,
        rolling=True,
        model_name="GAMMA_ML",
    )
    assert count_files(data_dir, glob_pattern="L2M_GAMMA_ML.ROLL1MIN.*.nc", recursive=True) == 2
