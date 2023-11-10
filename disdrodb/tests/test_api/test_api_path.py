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
"""Test DISDRODB path."""
import datetime
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.api.path import (
    define_campaign_dir,
    define_l0a_station_dir,
    define_l0b_station_dir,
    define_l0b_filepath,
    define_l0a_filepath,
)
 

PROCESSED_FOLDER_WINDOWS = "\\DISDRODB\\Processed"
PROCESSED_FOLDER_LINUX = "/DISDRODB/Processed"

    
@pytest.mark.parametrize("processed_folder", [PROCESSED_FOLDER_WINDOWS, PROCESSED_FOLDER_LINUX])
def test_define_l0a_station_dir(processed_folder):
    res = (
        define_l0a_station_dir(processed_folder, "STATION_NAME")
        .replace(processed_folder, "")
        .replace("\\", "")
        .replace("/", "")
    )
    assert res == "L0ASTATION_NAME"


@pytest.mark.parametrize("processed_folder", [PROCESSED_FOLDER_WINDOWS, PROCESSED_FOLDER_LINUX])
def test_define_l0b_station_dir(processed_folder):
    res = (
        define_l0b_station_dir(processed_folder, "STATION_NAME")
        .replace(processed_folder, "")
        .replace("\\", "")
        .replace("/", "")
    )
    assert res == "L0BSTATION_NAME"


def test_define_l0a_filepath(tmp_path):
    from disdrodb.l0.standards import PRODUCT_VERSION

    # Set variables
    product = "L0A"
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
    start_date_str = start_date.strftime("%Y%m%d%H%M%S")
    end_date_str = end_date.strftime("%Y%m%d%H%M%S")

    # Set paths
    processed_dir = define_campaign_dir(
        base_dir=base_dir, product=product, data_source=data_source, campaign_name=campaign_name
    )

    # Create dataframe
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})

    # Test the function
    res = define_l0a_filepath(df, processed_dir, station_name)

    # Define expected results
    expected_name = (
        f"{product}.{campaign_name.upper()}.{station_name}.s{start_date_str}.e{end_date_str}.{PRODUCT_VERSION}.parquet"
    )
    expected_path = os.path.join(processed_dir, product, station_name, expected_name)
    assert res == expected_path


def test_define_l0b_filepath(tmp_path):
    from disdrodb.l0.standards import PRODUCT_VERSION

    # Set variables

    product = "L0B"
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
    start_date_str = start_date.strftime("%Y%m%d%H%M%S")
    end_date_str = end_date.strftime("%Y%m%d%H%M%S")

    # Set paths
    processed_dir = define_campaign_dir(
        base_dir=base_dir, product=product, data_source=data_source, campaign_name=campaign_name
    )

    # Create xarray object
    timesteps = pd.date_range(start=start_date, end=end_date)
    data = np.zeros(timesteps.shape)
    ds = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": pd.date_range(start=start_date, end=end_date)},
    )

    # Test the function
    res = define_l0b_filepath(ds, processed_dir, station_name)

    # Define expected results
    expected_name = (
        f"{product}.{campaign_name.upper()}.{station_name}.s{start_date_str}.e{end_date_str}.{PRODUCT_VERSION}.nc"
    )
    expected_path = os.path.join(processed_dir, product, station_name, expected_name)
    assert res == expected_path
