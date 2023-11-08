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
"""Test DISDRDB checks of the campaign directory."""

import os
 
import pytest
 
from disdrodb.l0.check_campaign_directories import (
    _check_raw_dir_is_a_directory, 
    check_raw_dir,
    check_processed_dir,
)

from disdrodb import __root_path__
from disdrodb.tests.conftest import create_fake_metadata_file

TEST_DATA_DIR = os.path.join(__root_path__, "disdrodb", "tests", "data")
  
    
def test_check_raw_dir_is_a_directory(tmp_path):
    base_dir = tmp_path / "DISDRODB"
    station_name = "station_1"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    metadata_dict = {}
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        metadata_dict=metadata_dict,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    _check_raw_dir_is_a_directory(str(base_dir))
    
    
def test_check_raw_dir():
    # Set variables
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"

    # Set paths
    raw_dir = os.path.join(
        TEST_DATA_DIR,
        "test_dir_structure",
        "DISDRODB",
        "Raw",
        data_source,
        campaign_name,
    )

    assert check_raw_dir(raw_dir) == raw_dir
    

def test_check_processed_dir(tmp_path):
    # Check correct path
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "DATA_SOURCE" / "CAMPAIGN_NAME"
    processed_dir.mkdir(parents=True, exist_ok=True)

    assert check_processed_dir(str(processed_dir)) == str(processed_dir)

    # Check wrong type raises error
    with pytest.raises(TypeError):
        check_processed_dir(1)

    # Check wrong path (Raw)
    processed_dir = tmp_path / "DISDRODB" / "Raw" / "DATA_SOURCE" / "CAMPAIGN_NAME"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (only data_source)
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "DATA_SOURCE"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (only Processed)
    processed_dir = tmp_path / "DISDRODB" / "Processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (station_dir)
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "DATA_SOURCE" / "CAMPAIGN_NAME" / "data" / "station_name"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (lowercase data_source)
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "data_source" / "CAMPAIGN_NAME"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))

    # Check wrong path (lowercase data_source)
    processed_dir = tmp_path / "DISDRODB" / "Processed" / "DATA_SOURCE" / "campaign_name"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        check_processed_dir(str(processed_dir))
        
    
