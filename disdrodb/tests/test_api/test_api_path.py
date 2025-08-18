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

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.api.path import (
    define_l0a_filename,
    define_l0b_filename,
    define_l0c_filename,
)
from disdrodb.constants import ARCHIVE_VERSION

# PROCESSED_FOLDER_WINDOWS = "\\DISDRODB\\RAW"
# PROCESSED_FOLDER_LINUX = "/DISDRODB/RAW"


# @pytest.mark.parametrize("processed_folder", [PROCESSED_FOLDER_WINDOWS, PROCESSED_FOLDER_LINUX])
# def test_define_l0a_station_dir(processed_folder):
#     res = (
#         define_l0a_station_dir(processed_folder, "STATION_NAME")
#         .replace(processed_folder, "")
#         .replace("\\", "")
#         .replace("/", "")
#     )
#     assert res == "L0ASTATION_NAME"


def test_define_l0a_filename():
    # Set variables
    product = "L0A"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)

    # Create dataframe
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})

    # Define expected results
    expected_name = f"{product}.CAMPAIGN_NAME.STATION_NAME.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.parquet"

    # Test the function
    res = define_l0a_filename(df, campaign_name, station_name)
    assert res == expected_name


@pytest.mark.parametrize("product", ["L0B", "L0C"])
def test_define_l0b_filename(product):
    # Set variables
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
    sample_interval = 10
    sample_interval_str = "10S"
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)

    # Create xarray dataset
    timesteps = pd.date_range(start=start_date, end=end_date)
    data = np.zeros(timesteps.shape)
    ds = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": pd.date_range(start=start_date, end=end_date), "sample_interval": sample_interval},
    ).to_dataset(name="dummy")

    # Define expected results
    # TODO: MODIFY !
    if product == "L0B":
        expected_name = f"{product}.CAMPAIGN_NAME.STATION_NAME.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.nc"
    else:
        expected_name = f"{product}.{sample_interval_str}.CAMPAIGN_NAME.STATION_NAME.s20190326000000.e20210208000000.{ARCHIVE_VERSION}.nc"  # noqa: E501

    # Test the function
    define_filename_func = define_l0b_filename if product == "L0B" else define_l0c_filename
    res = define_filename_func(ds, campaign_name, station_name)
    assert res == expected_name
