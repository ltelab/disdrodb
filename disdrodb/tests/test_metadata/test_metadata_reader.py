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
"""Test DISDRODB metadata reader."""

import pytest

from disdrodb.metadata.reader import read_station_metadata
from disdrodb.tests.conftest import create_fake_metadata_file


@pytest.mark.parametrize("product", ["RAW", "L0A", "L0B"])
def test_read_station_metadata(tmp_path, product):
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "station_name"

    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    metadata_dict = read_station_metadata(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    assert isinstance(metadata_dict, dict)
