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
"""Test Metadata Manipulation Tools."""


from disdrodb.metadata.manipulation import (
    add_missing_metadata_keys,
    remove_invalid_metadata_keys,
)


def test_remove_invalid_metadata_keys():
    metadata = {"data_source": "valid", "BLABLA": "invalid"}
    expected = {"data_source": "valid"}
    result = remove_invalid_metadata_keys(metadata)
    assert result == expected


def test_add_missing_metadata_keys():
    metadata = {"data_source": "valid"}
    result = add_missing_metadata_keys(metadata)
    # Assert that the valid key is still there
    assert result["data_source"] == "valid"
    # Assert that the missing keys are added
    assert "campaign_name" in result
