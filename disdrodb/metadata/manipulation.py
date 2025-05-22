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
    from disdrodb.metadata.standards import METADATA_KEYS

    list_metadata_keys = METADATA_KEYS
    metadata = {k: metadata[k] for k in list_metadata_keys}
    return metadata
