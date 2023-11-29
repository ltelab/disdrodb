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

from disdrodb.api.create_directories import (
    _check_campaign_name_consistency,
    _check_data_source_consistency,
    _copy_station_metadata,
    create_directory_structure,
    create_initial_directory_structure,
    create_initial_station_structure,
    create_issue_directory,
    create_metadata_directory,
    create_test_archive,
)
from disdrodb.api.path import (
    define_campaign_dir,
    define_issue_filepath,
    define_metadata_dir,
    define_metadata_filepath,
    define_station_dir,
)
from disdrodb.tests.conftest import create_fake_metadata_directory, create_fake_metadata_file, create_fake_raw_data_file
from disdrodb.utils.yaml import read_yaml


@pytest.mark.parametrize("product", ["L0A", "L0B"])
def test_create_initial_directory_structure(tmp_path, mocker, product):
    # Define station info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"
    raw_dir = define_campaign_dir(
        base_dir=base_dir, product="RAW", data_source=data_source, campaign_name=campaign_name
    )

    processed_dir = define_campaign_dir(
        base_dir=base_dir, product=product, data_source=data_source, campaign_name=campaign_name
    )

    dst_metadata_dir = define_metadata_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    dst_station_dir = define_station_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    dst_metadata_filepath = define_metadata_filepath(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )

    # Create station Raw directory structure (with fake data)
    # - Add fake raw data file
    _ = create_fake_raw_data_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # - Add fake metadata
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # def mock_check_metadata_compliance(data_source, campaign_name, station_name, base_dir=None):
    #     return None
    # from disdrodb.metadata import check_metadata
    # check_metadata.check_metadata_compliance = mock_check_metadata_compliance

    # Mock to pass metadata checks
    mocker.patch("disdrodb.metadata.check_metadata.check_metadata_compliance", return_value=None)

    # Test that if station_name is unexisting in data, raise error
    with pytest.raises(ValueError):
        create_initial_directory_structure(
            product=product,
            force=False,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            station_name="INEXISTENT_STATION",
        )

    # Execute create_initial_directory_structure
    create_initial_directory_structure(
        product=product,
        force=False,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        station_name=station_name,
    )

    # Check create default metadata/issue files in raw_dir !
    # - TODO

    # Test product, metadata and station directories have been created
    assert os.path.exists(dst_station_dir) and os.path.isdir(dst_station_dir)
    assert os.path.exists(dst_metadata_dir) and os.path.isdir(dst_metadata_dir)
    # Test it copied the metadata from RAW
    assert os.path.exists(dst_metadata_filepath) and os.path.isfile(dst_metadata_filepath)
    os.remove(dst_metadata_filepath)

    # Test raise error if already data in L0A (if force=False)
    product_filepath = create_fake_raw_data_file(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    with pytest.raises(ValueError):
        create_initial_directory_structure(
            product=product,
            force=False,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            station_name=station_name,
        )
    assert os.path.exists(product_filepath)

    # Test delete file if already data in L0A (if force=True)
    create_initial_directory_structure(
        product=product,
        force=True,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        station_name=station_name,
    )
    assert not os.path.exists(product_filepath)
    assert os.path.exists(dst_station_dir) and os.path.isdir(dst_station_dir)
    assert os.path.exists(dst_metadata_dir) and os.path.isdir(dst_metadata_dir)
    assert os.path.exists(dst_metadata_filepath) and os.path.isfile(dst_metadata_filepath)


def test_create_directory_structure(tmp_path, mocker):
    start_product = "L0A"
    dst_product = "L0B"
    # Define station info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_1"

    processed_dir = define_campaign_dir(
        base_dir=base_dir, product=start_product, data_source=data_source, campaign_name=campaign_name
    )

    # Test raise error without data
    with pytest.raises(ValueError):
        create_directory_structure(
            product=dst_product,
            force=False,
            processed_dir=processed_dir,
            station_name=station_name,
        )

    # Add fake file
    _ = create_fake_raw_data_file(
        base_dir=base_dir,
        product=start_product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Test raise error without metadata file
    with pytest.raises(ValueError):
        create_directory_structure(
            product=dst_product,
            force=False,
            processed_dir=processed_dir,
            station_name=station_name,
        )

    # Add metadata
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product=start_product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Execute create_directory_structure
    create_directory_structure(
        processed_dir=processed_dir, product=dst_product, station_name=station_name, force=False, verbose=False
    )

    # Test product directory has been created
    dst_station_dir = os.path.join(processed_dir, dst_product)
    assert os.path.exists(dst_station_dir) and os.path.isdir(dst_station_dir)

    # Test raise error if already data in dst_product (if force=False)
    dst_product_file_filepath = create_fake_raw_data_file(
        base_dir=base_dir,
        product=dst_product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    with pytest.raises(ValueError):
        create_directory_structure(
            product=dst_product,
            force=False,
            processed_dir=processed_dir,
            station_name=station_name,
        )
    assert os.path.exists(dst_product_file_filepath)

    # Test delete file if already data in L0A (if force=True)
    create_directory_structure(
        product=dst_product,
        force=True,
        processed_dir=processed_dir,
        station_name=station_name,
    )
    assert not os.path.exists(dst_product_file_filepath)
    assert os.path.exists(dst_station_dir) and os.path.isdir(dst_station_dir)

    # Test raise error if bad station_name
    with pytest.raises(ValueError):
        create_directory_structure(
            product=dst_product,
            force=False,
            processed_dir=processed_dir,
            station_name="INEXISTENT_STATION",
        )


def test_create_initial_station_structure(tmp_path):
    """Check creation of station initial files and directories."""
    base_dir = tmp_path / "DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "station_name"
    # Create initial station structure
    create_initial_station_structure(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Check metadata and issue files have been created
    metadata_filepath = define_metadata_filepath(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )
    issue_filepath = define_issue_filepath(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    assert os.path.exists(metadata_filepath)
    assert os.path.exists(issue_filepath)

    # Check that if called once again, it raise error
    with pytest.raises(ValueError):
        create_initial_station_structure(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )


def test_create_test_archive(tmp_path):
    """Check creation of test archive."""
    base_dir = tmp_path / "base" / "DISDRODB"
    test_base_dir = tmp_path / "test" / "DISDRODB"

    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "station_name"
    # Create initial station structure
    create_initial_station_structure(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    create_test_archive(
        test_base_dir=test_base_dir,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Check metadata and issue files have been created
    metadata_filepath = define_metadata_filepath(
        base_dir=test_base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )
    issue_filepath = define_issue_filepath(
        base_dir=test_base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    assert os.path.exists(metadata_filepath)
    assert os.path.exists(issue_filepath)

    # Check that if called once again, it raise error
    with pytest.raises(ValueError):
        create_initial_station_structure(
            base_dir=base_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )


def test_check_campaign_name_consistency(tmp_path):
    base_dir = tmp_path / "DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"

    raw_dir = define_campaign_dir(
        base_dir=base_dir, product="RAW", data_source=data_source, campaign_name=campaign_name
    )

    # Test when consistent
    processed_dir = define_campaign_dir(
        base_dir=base_dir, product="L0A", data_source=data_source, campaign_name=campaign_name
    )
    assert _check_campaign_name_consistency(raw_dir, processed_dir) == campaign_name

    # Test when is not consistent
    processed_dir = define_campaign_dir(
        base_dir=base_dir, product="L0A", data_source=data_source, campaign_name="ANOTHER_CAMPAIGN_NAME"
    )
    with pytest.raises(ValueError):
        _check_campaign_name_consistency(raw_dir, processed_dir)


def test_check_data_source_consistency(tmp_path):
    base_dir = tmp_path / "DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    raw_dir = define_campaign_dir(
        base_dir=base_dir, product="RAW", data_source=data_source, campaign_name=campaign_name
    )

    # Test when consistent
    processed_dir = define_campaign_dir(
        base_dir=base_dir, product="L0A", data_source=data_source, campaign_name=campaign_name
    )

    assert _check_data_source_consistency(raw_dir, processed_dir) == data_source

    # Test when is not consistent
    processed_dir = define_campaign_dir(
        base_dir=base_dir, product="L0A", data_source="ANOTHER_DATA_SOURCE", campaign_name=campaign_name
    )
    with pytest.raises(ValueError):
        _check_data_source_consistency(raw_dir, processed_dir)


def test_copy_station_metadata(tmp_path):
    base_dir = tmp_path / "DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "STATION_NAME"

    raw_dir = define_campaign_dir(
        base_dir=base_dir, product="RAW", data_source=data_source, campaign_name=campaign_name
    )
    processed_dir = define_campaign_dir(
        base_dir=base_dir, product="L0A", data_source=data_source, campaign_name=campaign_name
    )
    dst_metadata_filepath = define_metadata_filepath(
        base_dir=base_dir,
        product="L0A",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )

    # Create fake metadata directory in RAW
    _ = create_fake_metadata_directory(
        base_dir=base_dir, product="RAW", data_source=data_source, campaign_name=campaign_name
    )

    # Test raise error if no data
    with pytest.raises(ValueError):
        _copy_station_metadata(raw_dir=raw_dir, processed_dir=processed_dir, station_name=station_name)

    # Create fake metadata file
    raw_metadata_filepath = create_fake_metadata_file(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Test raise error if no destination metadata directory
    with pytest.raises(ValueError):
        _copy_station_metadata(raw_dir=raw_dir, processed_dir=processed_dir, station_name=station_name)

    # Copy metadata
    _ = create_metadata_directory(
        base_dir=base_dir, product="L0A", data_source=data_source, campaign_name=campaign_name
    )

    _copy_station_metadata(raw_dir=raw_dir, processed_dir=processed_dir, station_name=station_name)

    # Ensure metadata file has been copied
    assert os.path.exists(dst_metadata_filepath)

    # Read both files and check are equally
    src_dict = read_yaml(raw_metadata_filepath)
    dst_dict = read_yaml(dst_metadata_filepath)
    assert src_dict == dst_dict


def test_create_issue_directory(tmp_path):
    base_dir = tmp_path / "DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"

    issue_dir = create_issue_directory(base_dir, data_source=data_source, campaign_name=campaign_name)
    assert os.path.isdir(issue_dir)
