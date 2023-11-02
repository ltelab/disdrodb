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

import netCDF4 as nc
import numpy as np
import xarray as xr
import yaml

from disdrodb.l0.l0b_nc_concat import _concatenate_netcdf_files, run_disdrodb_l0b_concat


def create_dummy_netcdf_file(filename: str, data: tuple):
    # Create a new NetCDF file
    with nc.Dataset(filename, mode="w") as ds:
        # Define dimensions
        ds.createDimension("lat", size=10)
        ds.createDimension("lon", size=10)
        ds.createDimension("time", size=None)

        # Create variables
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        time_var = ds.createVariable("time", "f8", ("time",))
        data_var = ds.createVariable("rainfall_rate_32bit", "f4", ("time", "lat", "lon"))

        ds.setncattr("sensor_name", "OTT_Parsivel")

        # Assign variable attributes
        lat_var.long_name = "latitude"
        lat_var.units = "degrees_north"
        lon_var.long_name = "longitude"
        lon_var.units = "degrees_east"
        time_var.long_name = "time"
        time_var.units = "days since 2023-01-01"

        lat_var[:] = data[0]
        lon_var[:] = data[1]
        time_var[:] = data[2]
        data_var[:] = data[3]


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
        with open(file_path, "w") as f:
            yaml.dump({"station_name": station_name}, f)

    return subfolder_path


def test_concatenate_netcdf_files(tmp_path):
    # Assign data to variables
    lat_data = np.linspace(-90, 90, 10, dtype=np.float32)
    lon_data = np.linspace(-180, 180, 10, dtype=np.float32)

    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "test_station"

    time_data_1 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    data_1 = np.random.rand(len(time_data_1), len(lat_data), len(lon_data)).astype(np.float32)
    time_data_2 = np.array([3.0, 4.0, 5.0], dtype=np.float64)
    data_2 = np.random.rand(len(time_data_2), len(lat_data), len(lon_data)).astype(np.float32)

    folder_temp = create_fake_data_file(
        tmp_path, data_source=data_source, campaign_name=campaign_name, station_name=station_name
    )
    root_dir_path = os.path.join(tmp_path, "DISDRODB", "Processed", data_source, campaign_name)

    filename_1 = os.path.join(folder_temp, "test_1.nc")
    filename_2 = os.path.join(folder_temp, "test_2.nc")

    create_dummy_netcdf_file(filename=filename_1, data=(lat_data, lon_data, time_data_1, data_1))
    create_dummy_netcdf_file(filename=filename_2, data=(lat_data, lon_data, time_data_2, data_2))

    def mock_write_l0b(ds: xr.Dataset, fpath: str, force=False) -> None:
        ds.to_netcdf(fpath, engine="netcdf4")

    # Monkey patch the function
    from disdrodb.l0 import l0b_processing

    l0b_processing.write_l0b = mock_write_l0b

    _concatenate_netcdf_files(processed_dir=root_dir_path, station_name=station_name, remove=False, verbose=False)

    # read netcdf file
    path_to_file = glob.glob(os.path.join(root_dir_path, "L0B", "*.nc"))[0]

    ds = xr.open_dataset(path_to_file)

    assert len(ds.time.values) == 6


def test_run_disdrodb_l0b_concat(tmp_path):
    # This test relies on the configuration file of OTT_Parsivel. It is due to the fact that the mock
    # function is not able to be applied because of the way the function is called in the code (cmd).

    # Assign data to variables
    lat_data = np.linspace(-90, 90, 10, dtype=np.float32)
    lon_data = np.linspace(-180, 180, 10, dtype=np.float32)

    data_source = "data_source"
    campaign_name = "campaign_name"
    station_name = "test_station"

    time_data_1 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    data_1 = np.random.rand(len(time_data_1), len(lat_data), len(lon_data)).astype(np.float32)
    time_data_2 = np.array([3.0, 4.0, 5.0], dtype=np.float64)
    data_2 = np.random.rand(len(time_data_2), len(lat_data), len(lon_data)).astype(np.float32)

    folder_temp_1 = create_fake_data_file(
        tmp_path=tmp_path,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        with_metadata_file=True,
    )
    folder_temp_2 = create_fake_data_file(
        tmp_path=tmp_path,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        with_metadata_file=True,
    )

    base_dir = os.path.join(tmp_path, "DISDRODB")

    filename_1 = os.path.join(folder_temp_1, f"{station_name}_1.nc")
    filename_2 = os.path.join(folder_temp_2, f"{station_name}_2.nc")

    create_dummy_netcdf_file(filename=filename_1, data=(lat_data, lon_data, time_data_1, data_1))
    create_dummy_netcdf_file(filename=filename_2, data=(lat_data, lon_data, time_data_2, data_2))

    run_disdrodb_l0b_concat(
        base_dir=base_dir,
        data_sources=data_source,
        campaign_names=campaign_name,
        station_names=[station_name],
        remove_l0b=True,
        verbose=False,
    )

    assert glob.glob(os.path.join(base_dir, "Processed", data_source, campaign_name, "L0B", "*.nc"))[0]
