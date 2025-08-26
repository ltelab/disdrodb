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
"""Test DISDRODB L0 Directory Creation."""
import os

import pytest
from click.testing import CliRunner

from disdrodb.api.create_directories import (
    create_initial_station_structure,
    create_issue_directory,
    create_l0_directory_structure,
    create_logs_directory,
    create_product_directory,
    create_test_archive,
)
from disdrodb.api.path import (
    define_data_dir,
    define_issue_filepath,
    define_metadata_dir,
    define_metadata_filepath,
    define_station_dir,
)
from disdrodb.cli.disdrodb_initialize_station import disdrodb_initialize_station
from disdrodb.constants import ARCHIVE_VERSION
from disdrodb.tests.conftest import (
    create_fake_issue_file,
    create_fake_metadata_file,
    create_fake_raw_data_file,
)

# import pathlib
# tmp_path = pathlib.Path("/tmp/dummy4")


@pytest.mark.parametrize("product", ["L0A", "L0B"])
def test_create_l0_directory_structure(tmp_path, mocker, product):
    # Define station info
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"

    dst_metadata_dir = define_metadata_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    dst_station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
    )

    dst_metadata_filepath = define_metadata_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )

    # Create station RAW directory structure (with fake data)
    # - Add fake raw data file
    _ = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # - Add fake metadata
    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # - Add fake issue file
    _ = create_fake_issue_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Mock to pass metadata checks
    mocker.patch("disdrodb.metadata.checks.check_station_metadata", return_value=None)

    # Test that if station_name is unexisting in data, raise error
    with pytest.raises(ValueError):
        create_l0_directory_structure(
            metadata_archive_dir=metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="INEXISTENT_STATION",
            product=product,
            force=False,
        )

    # Execute create_l0_directory_structure
    data_dir = create_l0_directory_structure(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        force=False,
    )

    # Test product, metadata and station directories have been created
    assert os.path.exists(data_dir)
    assert os.path.isdir(data_dir)
    assert os.path.exists(dst_station_dir)
    assert os.path.isdir(dst_station_dir)
    assert os.path.exists(dst_metadata_dir)
    assert os.path.isdir(dst_metadata_dir)
    assert os.path.exists(dst_metadata_filepath)
    assert os.path.isfile(dst_metadata_filepath)

    # Test raise error if already data in L0A (if force=False)
    product_filepath = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    with pytest.raises(ValueError):
        create_l0_directory_structure(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
            force=False,
        )
    assert os.path.exists(product_filepath)

    # Test delete file if already data in L0A (if force=True)
    data_dir = create_l0_directory_structure(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        force=True,
    )
    assert not os.path.exists(product_filepath)
    assert os.path.exists(data_dir)
    assert os.path.isdir(data_dir)
    assert os.path.exists(dst_station_dir)
    assert os.path.isdir(dst_station_dir)
    assert os.path.exists(dst_metadata_dir)
    assert os.path.isdir(dst_metadata_dir)
    assert os.path.exists(dst_metadata_filepath)
    assert os.path.isfile(dst_metadata_filepath)


def test_create_product_directory(tmp_path):
    start_product = "L0A"
    dst_product = "L0B"
    # Define station info
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"
    # Define mandatory metadata
    metadata_dict = {}
    metadata_dict["sensor_name"] = "PARSIVEL"
    metadata_dict["reader"] = "GPM/IFLOODS"
    metadata_dict["measurement_interval"] = 60
    metadata_dict["deployment_status"] = "terminated"
    metadata_dict["deployment_mode"] = "land"

    # Test raise error without data
    with pytest.raises(ValueError):
        _ = create_product_directory(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=dst_product,
            force=False,
        )

    # Add fake file
    _ = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        product=start_product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Test raise error without metadata file
    with pytest.raises(ValueError):
        _ = create_product_directory(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=dst_product,
            force=False,
        )

    # Add metadata
    _ = create_fake_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_dict=metadata_dict,
    )

    # Execute create_product_directory
    data_dir = create_product_directory(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=dst_product,
        force=False,
    )

    # Test product data directory has been created
    expected_data_dir = define_data_dir(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=dst_product,
    )
    assert expected_data_dir == data_dir
    assert os.path.exists(data_dir)
    assert os.path.isdir(data_dir)

    # Test raise error if already data in dst_product (if force=False)
    dst_product_file_filepath = create_fake_raw_data_file(
        data_archive_dir=data_archive_dir,
        product=dst_product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    with pytest.raises(ValueError):
        _ = create_product_directory(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=dst_product,
            force=False,
        )
    assert os.path.exists(dst_product_file_filepath)

    # Test delete file if already data in L0A (if force=True)
    data_dir = create_product_directory(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=dst_product,
        force=True,
    )
    assert expected_data_dir == data_dir
    assert not os.path.exists(dst_product_file_filepath)
    assert os.path.exists(data_dir)
    assert os.path.isdir(data_dir)

    # Test raise error if bad station_name
    with pytest.raises(ValueError):
        _ = create_product_directory(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="INEXISTENT_STATION",
            product=dst_product,
            force=False,
        )


def test_create_initial_station_structure(tmp_path):
    """Check creation of station initial files and directories."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "station_name"

    # Create initial station structure
    create_initial_station_structure(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check metadata and issue files have been created
    metadata_filepath = define_metadata_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    issue_filepath = define_issue_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    assert os.path.exists(metadata_filepath)
    assert os.path.exists(issue_filepath)

    # Check that if called once again, it raise error
    with pytest.raises(ValueError):
        create_initial_station_structure(
            data_archive_dir=data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )


def test_create_initial_station_structure_cmd(tmp_path):
    """Check creation of station initial files and directories."""
    data_archive_dir = tmp_path / "data" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "station_name"

    # Invoke command in the terminal
    runner = CliRunner()
    runner.invoke(
        disdrodb_initialize_station,
        [
            data_source,
            campaign_name,
            station_name,
            "--data_archive_dir",
            str(data_archive_dir),
            "--metadata_archive_dir",
            str(metadata_archive_dir),
        ],
    )

    # Check metadata and issue files have been created
    metadata_filepath = define_metadata_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    issue_filepath = define_issue_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    assert os.path.exists(metadata_filepath)
    assert os.path.exists(issue_filepath)


def test_create_test_archive(tmp_path):
    """Check creation of test archive."""
    data_archive_dir = tmp_path / "original" / "DISDRODB"
    test_data_archive_dir = tmp_path / "test" / "DISDRODB"
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"

    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "STATION_NAME"

    # Check that if raise error if test_data_archive_dir equals data_archive_dir
    with pytest.raises(ValueError):
        create_test_archive(
            test_data_archive_dir=test_data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=test_data_archive_dir,
            metadata_archive_dir=metadata_archive_dir,
            force=True,
        )

    test_data_archive_dir = create_test_archive(
        test_data_archive_dir=test_data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        force=True,
    )
    assert os.path.exists(test_data_archive_dir)
    assert os.path.isdir(test_data_archive_dir)


def test_create_issue_directory(tmp_path):
    metadata_archive_dir = tmp_path / "metadata" / "DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"

    issue_dir = create_issue_directory(metadata_archive_dir, data_source=data_source, campaign_name=campaign_name)
    assert os.path.isdir(issue_dir)


# import pathlib
# tmp_path = pathlib.Path("/tmp/13")


class TestCreateLogsDirectory:
    """Tests for create_logs_directory."""

    def test_creates_logs_dir(self, tmp_path):
        """Test creates a new logs directory when it does not exist."""
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0A"

        logs_dir = create_logs_directory(
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=data_archive_dir,
        )

        # Path must exist and match expected structure
        expected = os.path.join(
            data_archive_dir,
            ARCHIVE_VERSION,
            data_source,
            campaign_name,
            "logs",
            "files",
            product,
            station_name,
        )
        assert logs_dir == str(expected)
        assert os.path.exists(expected)
        assert os.path.isdir(expected)

    def test_recreates_if_exists(self, tmp_path):
        """Test removes and recreates logs directory if it already exists."""
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0A"

        # First create logs dir
        logs_dir = create_logs_directory(
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=data_archive_dir,
        )

        # Put a file inside logs dir
        dummy_file = os.path.join(logs_dir, "old.log")
        with open(dummy_file, "w") as f:
            f.write("old content")
        assert os.path.exists(dummy_file)

        # Call again → should clear and recreate directory
        logs_dir2 = create_logs_directory(
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            data_archive_dir=data_archive_dir,
        )
        assert logs_dir == logs_dir2
        assert os.path.exists(logs_dir)
        # Old file must be gone
        assert not os.path.exists(dummy_file)
