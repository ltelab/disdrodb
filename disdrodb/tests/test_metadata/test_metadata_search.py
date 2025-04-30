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
"""Test DISDRODB API metadata utility."""
import os
import shutil

import pytest

import disdrodb
from disdrodb.api.path import define_station_dir
from disdrodb.metadata.search import (
    get_list_metadata,
)
from disdrodb.tests.conftest import (
    create_fake_metadata_file,
    create_fake_raw_data_file,
    get_default_product_kwargs,
)

# import pathlib
# tmp_path = pathlib.Path("/tmp/dummy7")


class TestGetListMetadata:

    def test_list_all_metadata(self, tmp_path):
        """Test return metadata filepaths of all stations in the DISDRODB Metadata Archive."""
        metadata_dir = tmp_path / "DISDRODB"

        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        expected_result = []

        # Test with one metadata file
        metadata_filepath = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
        )
        expected_result.append(metadata_filepath)
        assert expected_result == get_list_metadata(
            metadata_dir=metadata_dir,
            data_sources=data_source,
            campaign_names=campaign_name,
        )
        assert expected_result == get_list_metadata(
            metadata_dir=metadata_dir,
        )

        # Test with two metadata files
        metadata_filepath = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
        )
        expected_result.append(metadata_filepath)
        assert expected_result == get_list_metadata(
            metadata_dir=metadata_dir,
            data_sources=data_source,
            campaign_names=campaign_name,
        )
        assert expected_result == get_list_metadata(
            metadata_dir=metadata_dir,
        )

    def test_filtering_metadata(self, tmp_path):
        """Test return metadata filepaths of stations matching the filtering criteria."""
        metadata_dir = tmp_path / "DISDRODB"

        # Create metadata files with different data source
        metadata_filepath1 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source="DATA_SOURCE_1",
            campaign_name="CAMPAIGN_NAME_1",
            station_name="station_1",
        )
        metadata_filepath2 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source="DATA_SOURCE_2",
            campaign_name="CAMPAIGN_NAME_2",
            station_name="station_2",
        )

        # Check filtering works
        assert [metadata_filepath1] == get_list_metadata(
            metadata_dir=metadata_dir,
            data_sources="DATA_SOURCE_1",
        )
        assert [metadata_filepath1] == get_list_metadata(
            metadata_dir=metadata_dir,
            campaign_names="CAMPAIGN_NAME_1",
        )
        assert [metadata_filepath1] == get_list_metadata(
            metadata_dir=metadata_dir,
            station_names="station_1",
        )
        assert [metadata_filepath1, metadata_filepath2] == get_list_metadata(
            metadata_dir=metadata_dir,
            data_sources=["DATA_SOURCE_1", "DATA_SOURCE_2"],
        )

        # Check filtering with invalid filtering criteria
        assert [metadata_filepath1] == get_list_metadata(
            metadata_dir=metadata_dir,
            data_sources=["DATA_SOURCE_1", "INVALID_DATA_SOURCE"],
            invalid_fields_policy="ignore",
        )
        with pytest.raises(ValueError):
            get_list_metadata(
                metadata_dir=metadata_dir,
                data_sources=["DATA_SOURCE_1", "INVALID_DATA_SOURCE"],
                invalid_fields_policy="raise",
            )
        # Check raise error if no stations left
        with pytest.raises(ValueError):
            get_list_metadata(
                metadata_dir=metadata_dir,
                data_sources=["INVALID_DATA_SOURCE"],
                invalid_fields_policy="ignore",
            )

    def test_list_metadata_with_disdrodb_data_url(self, tmp_path):
        """Test return metadata filepaths of metadata with disdrodb_data_url specified."""
        metadata_dir = tmp_path / "DISDRODB"

        # Create metadata files without disdrodb_data_url
        metadata_dict = {"disdrodb_data_url": ""}
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        metadata_filepath1 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            metadata_dict=metadata_dict,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
        )

        # Create metadata files without disdrodb_data_url
        metadata_dict = {"disdrodb_data_url": None}
        metadata_filepath2 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            metadata_dict=metadata_dict,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
        )

        # Create metadata file with disdrodb_data_url
        metadata_dict = {"disdrodb_data_url": "valid_url"}
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        metadata_filepath3 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            metadata_dict=metadata_dict,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_3",
        )

        # Check metadata file list
        metadata_filepaths = get_list_metadata(
            metadata_dir=metadata_dir,
            available_data=True,
        )

        assert metadata_filepath1 not in metadata_filepaths
        assert metadata_filepath2 not in metadata_filepaths
        assert metadata_filepath3 in metadata_filepaths

    def test_list_metadata_with_raw_data(self, tmp_path):
        """Test return metadata filepaths of stations with raw data."""
        base_dir = tmp_path / "data" / "DISDRODB"
        metadata_dir = tmp_path / "metadata" / "DISDRODB"

        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        # Create metadata and data for one station
        metadata_filepath1 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
        )
        _ = create_fake_raw_data_file(
            base_dir=base_dir,
            product="RAW",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
            filename="test_data.txt",
        )

        # Create only metadata for another station
        metadata_filepath2 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
        )

        # Check metadata file list
        metadata_filepaths = get_list_metadata(
            metadata_dir=metadata_dir,
            base_dir=base_dir,
            product="RAW",
            available_data=True,
        )

        assert metadata_filepath1 in metadata_filepaths
        assert metadata_filepath2 not in metadata_filepaths

    @pytest.mark.parametrize("product", disdrodb.PRODUCTS)
    def test_list_metadata_with_product_data(self, tmp_path, product):
        """Test return metadata filepaths of stations with product data."""
        base_dir = tmp_path / "data" / "DISDRODB"
        metadata_dir = tmp_path / "metadata" / "DISDRODB"

        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        # Define required product arguments (using some defaults for testing)
        product_kwargs = get_default_product_kwargs(product)

        # Create metadata and data for one station
        metadata_filepath1 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
        )
        _ = create_fake_raw_data_file(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
            filename="test_data.nc",
            **product_kwargs,
        )

        # Create only metadata for another station
        metadata_filepath2 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
        )

        # Check metadata file list
        metadata_filepaths = get_list_metadata(
            metadata_dir=metadata_dir,
            base_dir=base_dir,
            available_data=True,
            product=product,
            **product_kwargs,
        )

        assert metadata_filepath1 in metadata_filepaths
        assert metadata_filepath2 not in metadata_filepaths

    def test_list_metadata_with_raw_data_directory(self, tmp_path):
        """Test return metadata filepaths of stations with product data."""
        base_dir = tmp_path / "data" / "DISDRODB"
        metadata_dir = tmp_path / "metadata" / "DISDRODB"

        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        product = "RAW"

        # Create metadata and data for two station
        metadata_filepath1 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
        )
        filepath1 = create_fake_raw_data_file(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
            filename="test_data.txt",
        )
        _ = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
        )
        filepath2 = create_fake_raw_data_file(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
            filename="test_data.txt",
        )
        # Remove files
        os.remove(filepath1)
        os.remove(filepath2)

        # Remove also station directory for station 2
        shutil.rmtree(os.path.dirname(filepath2))

        # Check only station 1 metadata filepath is returned if available_data=False (default)
        assert [metadata_filepath1] == get_list_metadata(
            metadata_dir=metadata_dir,
            base_dir=base_dir,
            available_data=False,
            product=product,
        )
        assert [metadata_filepath1] == get_list_metadata(
            metadata_dir=metadata_dir,
            base_dir=base_dir,
            product=product,
        )
        # Check no station is returned if available_data=True
        assert (
            get_list_metadata(
                metadata_dir=metadata_dir,
                base_dir=base_dir,
                product=product,
                available_data=True,
            )
            == []
        )

    @pytest.mark.parametrize("product", disdrodb.PRODUCTS)
    def test_list_metadata_with_product_directory(self, tmp_path, product):
        """Test return metadata filepaths of stations with product data."""
        base_dir = tmp_path / "data" / "DISDRODB"
        metadata_dir = tmp_path / "metadata" / "DISDRODB"

        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"

        # Define required product arguments (using some defaults for testing)
        product_kwargs = get_default_product_kwargs(product)

        # Create metadata and data for two station
        metadata_filepath1 = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
        )
        filepath1 = create_fake_raw_data_file(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_1",
            filename="test_data.nc",
            **product_kwargs,
        )
        _ = create_fake_metadata_file(
            metadata_dir=metadata_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
        )
        filepath2 = create_fake_raw_data_file(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
            filename="test_data.nc",
            **product_kwargs,
        )
        # Remove files
        os.remove(filepath1)
        os.remove(filepath2)

        # Remove also product station directory for station 2
        product_dir_station_dir = define_station_dir(
            base_dir=base_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name="station_2",
        )
        shutil.rmtree(product_dir_station_dir)

        # Check only station 1 metadata filepath is returned if available_data=False (default)
        assert [metadata_filepath1] == get_list_metadata(
            metadata_dir=metadata_dir,
            base_dir=base_dir,
            available_data=False,
            product=product,
            **product_kwargs,
        )
        assert [metadata_filepath1] == get_list_metadata(
            metadata_dir=metadata_dir,
            base_dir=base_dir,
            product=product,
            **product_kwargs,
        )
        # Check no station is returned if available_data=True
        assert (
            get_list_metadata(
                metadata_dir=metadata_dir,
                base_dir=base_dir,
                product=product,
                available_data=True,
                **product_kwargs,
            )
            == []
        )

    def test_empty_metadata_archive_case(self, tmp_path):
        """Test return empty list of raise error if no metadata available."""
        metadata_dir = tmp_path / "metadata" / "DISDRODB"

        # Create metadata file and remove it
        # - Initialize DISDRODB Metadata Archive
        metadata_filepath = create_fake_metadata_file(metadata_dir=metadata_dir)
        os.remove(metadata_filepath)

        # Check return empty list
        assert get_list_metadata(metadata_dir=metadata_dir) == []

        # Check raise error if raise_error_if_empty=True
        with pytest.raises(ValueError):
            get_list_metadata(metadata_dir=metadata_dir, raise_error_if_empty=True)

    def test_empty_data_archive_case(self, tmp_path):
        """Test return empty list of raise error if no data available."""
        base_dir = tmp_path / "data" / "DISDRODB"
        metadata_dir = tmp_path / "metadata" / "DISDRODB"

        # Create DISDRODB Metadata Archive
        _ = create_fake_metadata_file(metadata_dir=metadata_dir)

        # Create DISDRODB Data Archive
        # - Just initialize directory structure
        file_filepath = create_fake_raw_data_file(base_dir=base_dir, product="RAW", filename="test_data.txt")
        os.remove(file_filepath)
        file_filepath = create_fake_raw_data_file(base_dir=base_dir, product="L0C", filename="test_data.nc")
        os.remove(file_filepath)

        # Check return empty list
        assert get_list_metadata(metadata_dir=metadata_dir, base_dir=base_dir, product="RAW", available_data=True) == []
        assert get_list_metadata(metadata_dir=metadata_dir, base_dir=base_dir, product="L0C", available_data=True) == []

        with pytest.raises(ValueError):
            get_list_metadata(
                metadata_dir=metadata_dir,
                base_dir=base_dir,
                product="RAW",
                available_data=True,
                raise_error_if_empty=True,
            )

        with pytest.raises(ValueError):
            get_list_metadata(
                metadata_dir=metadata_dir,
                base_dir=base_dir,
                product="L0C",
                available_data=True,
                raise_error_if_empty=True,
            )
