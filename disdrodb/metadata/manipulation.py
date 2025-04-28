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
"""Metadata Manipulation Tools."""
import shutil

from disdrodb.api.io import available_stations
from disdrodb.api.path import define_metadata_filepath
from disdrodb.configs import get_base_dir


def remove_invalid_metadata_keys(metadata):
    """Remove invalid keys from the metadata dictionary."""
    from disdrodb.metadata.checks import get_metadata_invalid_keys

    invalid_keys = get_metadata_invalid_keys(metadata)
    for k in invalid_keys:
        _ = metadata.pop(k)
    return metadata


def add_missing_metadata_keys(metadata):
    """Add missing keys to the metadata dictionary."""
    from disdrodb.metadata.checks import get_metadata_missing_keys

    missing_keys = get_metadata_missing_keys(metadata)
    for k in missing_keys:
        metadata[k] = ""
    return metadata


def sort_metadata_dictionary(metadata):
    """Sort the keys of the metadata dictionary by ``valid_metadata_keys`` list order."""
    from disdrodb.metadata.standards import get_valid_metadata_keys

    list_metadata_keys = get_valid_metadata_keys()
    metadata = {k: metadata[k] for k in list_metadata_keys}
    return metadata


def update_processed_metadata():
    """Update metadata in the 'DISDRODB/Processed' directory."""
    base_dir = get_base_dir()
    # Retrieve list of all processed stations
    # --> (data_source, campaign_name, station_name)
    list_info = available_stations(
        product="L0B",
    )

    # Retrieve metadata filepaths
    list_src_dst_path = [
        (
            # Source
            define_metadata_filepath(
                product="RAW",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                base_dir=base_dir,
                check_exists=False,
            ),
            # Destination
            define_metadata_filepath(
                product="L0B",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                base_dir=base_dir,
                check_exists=False,
            ),
        )
        for data_source, campaign_name, station_name in list_info
    ]
    # Copy file from RAW directory to Processed directory
    _ = [shutil.copyfile(src_path, dst_path) for (src_path, dst_path) in list_src_dst_path]
