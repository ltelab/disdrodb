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

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.l0.l0_processing import run_l0b_concat
from disdrodb.l0.routines import run_disdrodb_l0b_concat
from disdrodb.utils.netcdf import xr_concat_datasets
from disdrodb.utils.yaml import write_yaml


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


def test_xr_concat_datasets(tmp_path):
    # from pathlib import Path
    # tmp_path = Path("/tmp/test8")
    # tmp_path.mkdir()

    # Write L0B files
    filepath_1 = os.path.join(tmp_path, "test_1.nc")
    filepath_2 = os.path.join(tmp_path, "test_2.nc")

    time_data_1 = np.array(pd.date_range(start="2023-01-01", periods=3, freq="D"))
    time_data_2 = np.array(pd.date_range(start="2023-01-04", periods=3, freq="D"))

    create_dummy_l0b_file(filepath=filepath_1, time=time_data_1)
    create_dummy_l0b_file(filepath=filepath_2, time=time_data_2)

    # Check with file in correct orders
    fpaths = [filepath_1, filepath_2]
    ds = xr_concat_datasets(fpaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), np.concatenate((time_data_1, time_data_2)).astype(float))

    # Check with file in reverse orders
    fpaths = [filepath_2, filepath_1]
    ds = xr_concat_datasets(fpaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), np.concatenate((time_data_1, time_data_2)).astype(float))


def test_xr_concat_completely_overlapped_datasets(tmp_path):
    # Write L0B files
    filepath_1 = os.path.join(tmp_path, "test_1.nc")
    filepath_2 = os.path.join(tmp_path, "test_2.nc")
    filepath_3 = os.path.join(tmp_path, "test_3.nc")

    time_data_1 = np.array(pd.date_range(start="2023-01-01", periods=6, freq="D"))
    time_data_2 = np.array(pd.date_range(start="2023-01-04", periods=3, freq="D"))

    create_dummy_l0b_file(filepath=filepath_1, time=time_data_1)
    create_dummy_l0b_file(filepath=filepath_2, time=time_data_2)
    create_dummy_l0b_file(filepath=filepath_3, time=time_data_2[::-1])

    # Check with file in correct orders
    fpaths = [filepath_1, filepath_2]
    ds = xr_concat_datasets(fpaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), time_data_1.astype(float))

    # Check with file in reverse orders
    fpaths = [filepath_2, filepath_1]
    ds = xr_concat_datasets(fpaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), time_data_1.astype(float))

    # Check if completely overlapped but reversed order
    fpaths = [filepath_2, filepath_3]
    ds = xr_concat_datasets(fpaths)
    time_values = ds["time"].values
    assert len(time_values) == 3
    np.testing.assert_allclose(time_values.astype(float), time_data_2.astype(float))


def test_xr_concat_completely_partial_overlapped_datasets(tmp_path):
    # Write L0B files
    filepath_1 = os.path.join(tmp_path, "test_1.nc")
    filepath_2 = os.path.join(tmp_path, "test_2.nc")

    time_data_1 = np.array(pd.date_range(start="2023-01-01", periods=4, freq="D"))
    time_data_2 = np.array(pd.date_range(start="2023-01-04", periods=3, freq="D"))

    unique_time_data = np.sort(np.unique(np.concatenate((time_data_1, time_data_2))))

    create_dummy_l0b_file(filepath=filepath_1, time=time_data_1)
    create_dummy_l0b_file(filepath=filepath_2, time=time_data_2)

    # Check with file in correct orders
    fpaths = [filepath_1, filepath_2]
    ds = xr_concat_datasets(fpaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), unique_time_data.astype(float))

    # Check with file in reverse orders
    fpaths = [filepath_2, filepath_1]
    ds = xr_concat_datasets(fpaths)
    time_values = ds["time"].values
    assert len(time_values) == 6
    np.testing.assert_allclose(time_values.astype(float), unique_time_data.astype(float))


def create_fake_data_file(tmp_path, data_source, campaign_name, station_name="", with_metadata_file=False):
    subfolder_path = tmp_path / "DISDRODB" / "Processed" / data_source / campaign_name / "L0B" / station_name
    if not os.path.exists(subfolder_path):
        subfolder_path.mkdir(parents=True)
    assert os.path.exists(subfolder_path)

    if with_metadata_file:
        metedata_folder_path = tmp_path / "DISDRODB" / "Processed" / data_source / campaign_name / "metadata"
        if not os.path.exists(metedata_folder_path):
            metedata_folder_path.mkdir(parents=True)
        assert os.path.exists(metedata_folder_path)

        file_path = os.path.join(metedata_folder_path, f"{station_name}.yml")
        write_yaml({"station_name": station_name}, file_path)

    return subfolder_path


def test_run_l0b_concat_station(tmp_path):
    # Define station info
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "test_station"
    root_dir_path = os.path.join(tmp_path, "DISDRODB", "Processed", data_source, campaign_name)

    # Define fake L0B directory structure
    folder_temp = create_fake_data_file(
        tmp_path,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        with_metadata_file=False,
    )

    # Add dummy L0B files
    filepath_1 = os.path.join(folder_temp, "test_1.nc")
    filepath_2 = os.path.join(folder_temp, "test_2.nc")

    time_data_1 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    time_data_2 = np.array([3.0, 4.0, 5.0], dtype=np.float64)

    create_dummy_l0b_file(filepath=filepath_1, time=time_data_1)
    create_dummy_l0b_file(filepath=filepath_2, time=time_data_2)

    # Monkey patch the write_l0b function
    def mock_write_l0b(ds: xr.Dataset, fpath: str, force=False) -> None:
        ds.to_netcdf(fpath, engine="netcdf4")

    from disdrodb.l0 import l0b_processing

    l0b_processing.write_l0b = mock_write_l0b

    # Run concatenation command
    run_l0b_concat(processed_dir=root_dir_path, station_name=station_name, remove=False, verbose=False)

    # Read concatenated netCDF file
    path_to_file = glob.glob(os.path.join(root_dir_path, "L0B", "*.nc"))[0]
    ds = xr.open_dataset(path_to_file)
    assert len(ds["time"].values) == 6


def test_run_disdrodb_l0b_concat(tmp_path):
    # Define stations info
    base_dir = os.path.join(tmp_path, "DISDRODB")
    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name1 = "test_station_1"
    station_name2 = "test_station_2"

    # Define fake L0B directory structure for the two stations
    folder_temp_1 = create_fake_data_file(
        tmp_path=tmp_path,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name1,
        with_metadata_file=True,
    )
    folder_temp_2 = create_fake_data_file(
        tmp_path=tmp_path,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name2,
        with_metadata_file=True,
    )

    # Add dummy L0B files for two stations
    filepath_1 = os.path.join(folder_temp_1, f"{station_name1}_file.nc")
    filepath_2 = os.path.join(folder_temp_2, f"{station_name2}_file.nc")

    time_data_1 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    time_data_2 = np.array([3.0, 4.0, 5.0], dtype=np.float64)

    create_dummy_l0b_file(filepath=filepath_1, time=time_data_1)
    create_dummy_l0b_file(filepath=filepath_2, time=time_data_2)

    # Run concatenation command
    run_disdrodb_l0b_concat(
        base_dir=base_dir,
        data_sources=data_source,
        campaign_names=campaign_name,
        station_names=[station_name1, station_name2],
        remove_l0b=True,
        verbose=False,
    )

    # Assert the presence of 2 concatenated netcdf files (one for each station)
    list_files = glob.glob(os.path.join(base_dir, "Processed", data_source, campaign_name, "L0B", "*.nc"))
    assert len(list_files) == 2
