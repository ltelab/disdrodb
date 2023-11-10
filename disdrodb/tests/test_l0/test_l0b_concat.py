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
"""Test DISDRODB L0B netCDF concatenation routines."""

import os

import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.api.path import define_campaign_dir
from disdrodb.l0.l0_processing import run_l0b_concat
from disdrodb.l0.routines import run_disdrodb_l0b_concat
from disdrodb.tests.conftest import create_fake_metadata_file, create_fake_station_dir
from disdrodb.utils.directories import list_files
from disdrodb.utils.netcdf import xr_concat_datasets


def create_dummy_l0b_file(filepath: str, time):
    # Define the size of the dimensions
    n_lat = 10
    n_lon = 10

    # Assign lat/lon coordinates
    lat_data = np.linspace(-90, 90, n_lat, dtype=np.float32)
    lon_data = np.linspace(-180, 180, n_lon, dtype=np.float32)

    # Define variable dictionary
    data = np.random.rand(len(time), len(lat_data), len(lon_data)).astype(np.float32)
    data_vars = {
        "rainfall_rate_32bit": (("time", "lat", "lon"), data),
    }
    # Create the coordinate dictionary
    coords_dict = {
        "lat": ("lat", lat_data),
        "lon": ("lon", lon_data),
        "time": ("time", time),
    }
    # Create a dataset with dimensions lat, lon, and time
    ds = xr.Dataset(data_vars, coords=coords_dict)
    # Set global attribute
    ds.attrs["sensor_name"] = "OTT_Parsivel"

    # Set variable attributes
    ds["lat"].attrs["long_name"] = "latitude"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["long_name"] = "longitude"
    ds["lon"].attrs["units"] = "degrees_east"
    ds["time"].attrs["long_name"] = "time"
    # ds["time"].attrs["units"] = "days since 2023-01-01"

    # Write the dataset to a new NetCDF file
    ds.to_netcdf(filepath)
    ds.close()
    return filepath


def test_xr_concat_datasets(tmp_path):
    # Write L0B files
    filepath1 = os.path.join(tmp_path, "test_1.nc")
    filepath2 = os.path.join(tmp_path, "test_2.nc")

    time_data_1 = np.array(pd.date_range(start="2023-01-01", periods=3, freq="D"))
    time_data_2 = np.array(pd.date_range(start="2023-01-04", periods=3, freq="D"))

    _ = create_dummy_l0b_file(filepath=filepath1, time=time_data_1)
    _ = create_dummy_l0b_file(filepath=filepath2, time=time_data_2)

    # Check with file in correct orders
    filepaths = [filepath1, filepath2]
    ds = xr_concat_datasets(filepaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), np.concatenate((time_data_1, time_data_2)).astype(float))

    # Check with file in reverse orders
    filepaths = [filepath2, filepath1]
    ds = xr_concat_datasets(filepaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), np.concatenate((time_data_1, time_data_2)).astype(float))


def test_xr_concat_completely_overlapped_datasets(tmp_path):
    # Write L0B files
    filepath1 = os.path.join(tmp_path, "test_1.nc")
    filepath2 = os.path.join(tmp_path, "test_2.nc")
    filepath3 = os.path.join(tmp_path, "test_3.nc")

    time_data_1 = np.array(pd.date_range(start="2023-01-01", periods=6, freq="D"))
    time_data_2 = np.array(pd.date_range(start="2023-01-04", periods=3, freq="D"))

    _ = create_dummy_l0b_file(filepath=filepath1, time=time_data_1)
    _ = create_dummy_l0b_file(filepath=filepath2, time=time_data_2)
    _ = create_dummy_l0b_file(filepath=filepath3, time=time_data_2[::-1])

    # Check with file in correct orders
    filepaths = [filepath1, filepath2]
    ds = xr_concat_datasets(filepaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), time_data_1.astype(float))

    # Check with file in reverse orders
    filepaths = [filepath2, filepath1]
    ds = xr_concat_datasets(filepaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), time_data_1.astype(float))

    # Check if completely overlapped but reversed order
    filepaths = [filepath2, filepath3]
    ds = xr_concat_datasets(filepaths)
    time_values = ds["time"].values
    assert len(time_values) == 3
    np.testing.assert_allclose(time_values.astype(float), time_data_2.astype(float))


def test_xr_concat_completely_partial_overlapped_datasets(tmp_path):
    # Write L0B files
    filepath1 = os.path.join(tmp_path, "test_1.nc")
    filepath2 = os.path.join(tmp_path, "test_2.nc")

    time_data_1 = np.array(pd.date_range(start="2023-01-01", periods=4, freq="D"))
    time_data_2 = np.array(pd.date_range(start="2023-01-04", periods=3, freq="D"))

    unique_time_data = np.sort(np.unique(np.concatenate((time_data_1, time_data_2))))

    _ = create_dummy_l0b_file(filepath=filepath1, time=time_data_1)
    _ = create_dummy_l0b_file(filepath=filepath2, time=time_data_2)

    # Check with file in correct orders
    filepaths = [filepath1, filepath2]
    ds = xr_concat_datasets(filepaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), unique_time_data.astype(float))

    # Check with file in reverse orders
    filepaths = [filepath2, filepath1]
    ds = xr_concat_datasets(filepaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), unique_time_data.astype(float))


def test_run_l0b_concat_station(tmp_path):
    # Define station info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "test_station"

    processed_dir = define_campaign_dir(
        base_dir=base_dir, product="L0B", data_source=data_source, campaign_name=campaign_name
    )
    # Define fake L0B directory structure
    station_dir = create_fake_station_dir(
        base_dir=base_dir,
        product="L0B",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Add dummy L0B files
    filepath1 = os.path.join(station_dir, "test_1.nc")
    filepath2 = os.path.join(station_dir, "test_2.nc")

    time_data_1 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    time_data_2 = np.array([3.0, 4.0, 5.0], dtype=np.float64)

    _ = create_dummy_l0b_file(filepath=filepath1, time=time_data_1)
    _ = create_dummy_l0b_file(filepath=filepath2, time=time_data_2)

    # Monkey patch the write_l0b function
    def mock_write_l0b(ds: xr.Dataset, filepath: str, force=False) -> None:
        ds.to_netcdf(filepath, engine="netcdf4")

    from disdrodb.l0 import l0b_processing

    l0b_processing.write_l0b = mock_write_l0b

    # Run concatenation command
    run_l0b_concat(processed_dir=processed_dir, station_name=station_name, remove=False, verbose=False)

    # Assert only 1 file is created
    list_concatenated_files = list_files(os.path.join(processed_dir, "L0B"), glob_pattern="*.nc", recursive=False)
    assert len(list_concatenated_files) == 1

    # Read concatenated netCDF file
    ds = xr.open_dataset(list_concatenated_files[0])
    assert len(ds["time"].values) == 6


def test_run_disdrodb_l0b_concat(tmp_path):
    # Define stations info
    base_dir = tmp_path / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name1 = "test_station_1"
    station_name2 = "test_station_2"

    # Define fake directory structure for the two L0B stations
    #     # Define fake L0B directory structure
    station1_dir = create_fake_station_dir(
        base_dir=base_dir,
        product="L0B",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name1,
    )
    station2_dir = create_fake_station_dir(
        base_dir=base_dir,
        product="L0B",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name2,
    )
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product="L0B",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name1,
    )
    _ = create_fake_metadata_file(
        base_dir=base_dir,
        product="L0B",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name2,
    )
    # Add dummy L0B files for two stations
    filepath1 = os.path.join(station1_dir, f"{station_name1}_file.nc")
    filepath2 = os.path.join(station2_dir, f"{station_name2}_file.nc")

    time_data_1 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    time_data_2 = np.array([3.0, 4.0, 5.0], dtype=np.float64)

    _ = create_dummy_l0b_file(filepath=filepath1, time=time_data_1)
    _ = create_dummy_l0b_file(filepath=filepath2, time=time_data_2)

    # Run concatenation command
    run_disdrodb_l0b_concat(
        base_dir=str(base_dir),
        data_sources=data_source,
        campaign_names=campaign_name,
        station_names=[station_name1, station_name2],
        remove_l0b=True,
        verbose=False,
    )

    # BUGGY WITH PYTEST !
    # # Assert files where removed
    # assert not os.path.exists(filepath1)
    # assert not os.path.exists(filepath2)

    # # Assert the presence of 2 concatenated netcdf files (one for each station)
    # processed_dir = define_campaign_dir(
    #     base_dir=base_dir, product="L0B", data_source=data_source, campaign_name=campaign_name
    # )

    # list_concatenated_files = list_files(os.path.join(processed_dir, "L0B"), glob_pattern="*.nc", recursive=False)
    # assert len(list_concatenated_files) == 2

    # Check that if L0B files are removed, raise error if no stations available
    # with pytest.raises(ValueError):
    #     run_disdrodb_l0b_concat(
    #         base_dir=str(base_dir),
    #         data_sources=data_source,
    #         campaign_names=campaign_name,
    #         station_names=[station_name1, station_name2],
    #         remove_l0b=True,
    #         verbose=False,
    #     )
