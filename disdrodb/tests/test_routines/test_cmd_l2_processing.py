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

from disdrodb import __root_path__
from disdrodb.api.path import define_data_dir
from disdrodb.cli.disdrodb_run_l2e import disdrodb_run_l2e
from disdrodb.cli.disdrodb_run_l2e_station import disdrodb_run_l2e_station
from disdrodb.cli.disdrodb_run_l2m import disdrodb_run_l2m
from disdrodb.cli.disdrodb_run_l2m_station import disdrodb_run_l2m_station
from disdrodb.constants import ARCHIVE_VERSION
from disdrodb.routines import (
    run_l2e,
    run_l2e_station,
    run_l2m,
    run_l2m_station,
)
from disdrodb.utils.directories import count_files

TEST_DATA_L1_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "test_data_l1")
TEST_DATA_L2E_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data", "test_data_l2e")

DATA_SOURCE = "EPFL"
CAMPAIGN_NAME = "HYMEX_LTE_SOP2"
STATION_NAME = "10"
DEBUGGING_MODE = True
VERBOSE = True
FORCE = False


# from disdrodb.metadata.download import download_metadata_archive
# import pathlib
# tmp_path = pathlib.Path("/tmp/13")
# test_data_archive_dir = os.path.join(tmp_path, "DISDRODB")
# dst_dir = os.path.join(test_data_archive_dir, ARCHIVE_VERSION)
# shutil.copytree(TEST_DATA_L1_DIR, dst_dir, dirs_exist_ok=True)
# # shutil.copytree(TEST_DATA_L2E_DIR, dst_dir, dirs_exist_ok=True)
# parallel=False
# VERBOSE=True
# test_metadata_archive_dir = "/home/ghiggi/Projects/DISDRODB-METADATA/DISDRODB"
# test_metadata_archive_dir = download_metadata_archive(tmp_path / "original_metadata_archive_repo")
# os.environ["PYTEST_CURRENT_TEST"] = "1"


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l2e_station(tmp_path, disdrodb_metadata_archive_dir, parallel, cli):
    """Test the disdrodb_run_l2e_station command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_data_archive_dir / ARCHIVE_VERSION
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
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        run_l2e_station(
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
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

    # Check products at different temporal resolutions are produced
    temporal_resolutions = ["1MIN", "10MIN", "ROLL2MIN"]
    for temporal_resolution in temporal_resolutions:
        data_dir = define_data_dir(
            data_archive_dir=test_data_archive_dir,
            product="L2E",
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            temporal_resolution=temporal_resolution,
        )
        assert count_files(data_dir, glob_pattern=f"L2E.{temporal_resolution}.*.nc", recursive=True) == 2


@pytest.mark.parametrize("cli", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_disdrodb_run_l2m_station(tmp_path, disdrodb_metadata_archive_dir, parallel, cli):
    """Test the disdrodb_run_l2m_station command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_data_archive_dir / ARCHIVE_VERSION
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
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
            ],
        )
    else:
        run_l2m_station(
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
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

    # Check products are produced
    # - 10MIN GAMMA ML
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L2M",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        temporal_resolution="10MIN",
        model_name="GAMMA_ML",
    )
    assert count_files(data_dir, glob_pattern="L2M_GAMMA_ML.10MIN.*.nc", recursive=True) == 2

    # - 10MIN NGAMMA_GS_LOG_ND_MAE
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L2M",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        temporal_resolution="10MIN",
        model_name="NGAMMA_GS_LOG_ND_MAE",
    )
    assert count_files(data_dir, glob_pattern="L2M_NGAMMA_GS_LOG_ND_MAE.10MIN.*.nc", recursive=True) == 2


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_run_l2e(tmp_path, disdrodb_metadata_archive_dir, cli):
    """Test the disdrodb_run_l2e command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_data_archive_dir / ARCHIVE_VERSION
    shutil.copytree(TEST_DATA_L1_DIR, dst_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l2e,
            [
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
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
        run_l2e(
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
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

    # Check products at different temporal resolutions are produced
    temporal_resolutions = ["1MIN", "10MIN", "ROLL2MIN"]
    for temporal_resolution in temporal_resolutions:
        data_dir = define_data_dir(
            data_archive_dir=test_data_archive_dir,
            product="L2E",
            data_source=DATA_SOURCE,
            campaign_name=CAMPAIGN_NAME,
            station_name=STATION_NAME,
            temporal_resolution=temporal_resolution,
        )
        assert count_files(data_dir, glob_pattern=f"L2E.{temporal_resolution}.*.nc", recursive=True) == 2


@pytest.mark.parametrize("cli", [True, False])
def test_disdrodb_run_l2m(tmp_path, disdrodb_metadata_archive_dir, cli):
    """Test the disdrodb_run_l2m command."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    test_metadata_archive_dir = disdrodb_metadata_archive_dir  # fixture for the original DISDRODB Archive

    dst_dir = test_data_archive_dir / ARCHIVE_VERSION
    shutil.copytree(TEST_DATA_L2E_DIR, dst_dir)

    # Produce data
    if cli:
        runner = CliRunner()
        runner.invoke(
            disdrodb_run_l2m,
            [
                # DISDRODB root directories
                "--data_archive_dir",
                test_data_archive_dir,
                "--metadata_archive_dir",
                test_metadata_archive_dir,
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
        run_l2m(
            # DISDRODB root directories
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=test_metadata_archive_dir,
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

    # Check various products are produced
    # - 10MIN GAMMA ML
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L2M",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        temporal_resolution="10MIN",
        model_name="GAMMA_ML",
    )
    assert count_files(data_dir, glob_pattern="L2M_GAMMA_ML.10MIN.*.nc", recursive=True) == 2

    # - 10MIN NGAMMA_GS_LOG_ND_MAE
    data_dir = define_data_dir(
        data_archive_dir=test_data_archive_dir,
        product="L2M",
        data_source=DATA_SOURCE,
        campaign_name=CAMPAIGN_NAME,
        station_name=STATION_NAME,
        temporal_resolution="10MIN",
        model_name="NGAMMA_GS_LOG_ND_MAE",
    )
    assert count_files(data_dir, glob_pattern="L2M_NGAMMA_GS_LOG_ND_MAE.10MIN.*.nc", recursive=True) == 2
