#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:33:27 2022

@author: ghiggi
"""
# ------------------------------------------------------------------------------.
# DIVEN LPM
# COVERHEAD has no raw spectr
# "raw_drop_number",
# "raw_drop_concentration",
# "raw_drop_average_velocity",

import os
import glob
import xarray as xr

fpath = "/ltenas3/0_Data/DISDRODB/Raw/UK/DIVEN/data/CAIRNGORM/ncas-disdrometer-11_cairngorm_20170210_precipitation_v1.0.nc"
ds = xr.open_dataset(fpath)

ds

ds.data_vars

ds.coords

ds.attrs

# Check bins
ds.diameter
ds.fallspeed  # does not follow Thies documentation  !!!

# Get useful attributes for metadata
ds.latitude
ds.longitude

ds.attrs["averaging_interval"]
ds.attrs["sampling_interval"]
ds.attrs["instrument_model"]

# FULL CHECK
campaigns = ["DIVEN"]
base_dir = "/ltenas3/0_Data/DISDRODB/Raw/UK/"

fpath = "/ltenas3/0_Data/DISDRODB/Raw/UK/DIVEN/data/CAIRNGORM/ncas-disdrometer-11_cairngorm_20170210_precipitation_v1.0.nc"
ds_ref = xr.open_dataset(fpath)
vars_ref = set(ds_ref.data_vars)

list_stations_pattern = [
    os.path.join(base_dir, campaign, "data/*") for campaign in campaigns
]
list_stations_fpaths = [glob.glob(p) for p in list_stations_pattern]
list_stations_fpaths = [x for xs in list_stations_fpaths for x in xs]  # flatten list

list_station_file_example = [
    glob.glob(os.path.join(d, "*.nc"))[0] for d in list_stations_fpaths
]
assert len(list_station_file_example) == len(list_stations_fpaths)

# Look at variables changes
for f in list_station_file_example:
    print(f)
    ds = xr.open_dataset(f)
    var_set = set(ds.data_vars)
    diff_var = vars_ref.difference(var_set)
    if len(diff_var) != 0:
        print(diff_var)
    diff_var = var_set.difference(vars_ref)
    if len(diff_var) != 0:
        print(diff_var)

f = "/ltenas3/0_Data/DISDRODB/Raw/UK/DIVEN/data/COVERHEAD/ncas-disdrometer-14_coverhead_20170210_precipitation_v1.0.nc"
d = xr.open_dataset(f)
d.data_vars

# Look at attributes
for f in list_station_file_example:
    print(f)
    ds = xr.open_dataset(f)
    # print(ds.attrs["sampling_interval"])          # 1 minute
    print(ds.attrs["instrument_model"])  # 0.54 m^2


# Look at coordinate names
for f in list_station_file_example:
    print(f)
    ds = xr.open_dataset(f)
    print(list(ds.coords))

# Look at lat/lon/alt coordinates
for f in list_station_file_example:
    print(f)
    ds = xr.open_dataset(f)
    print(ds.lat.values)
    print(ds.lon.values)
    print(ds.alt.values)

for f in list_station_file_example:
    print(f)
    ds = xr.open_dataset(f)
    if ds.lat.values == -9999.0:
        print("MOBILE ?")  # MARCUS S1, MARCUS S2, MOSAIC M1, MOSAIC S3


# Encodings
from disdrodb.data_encodings import get_ARM_LPM_dict
from disdrodb.data_encodings import get_ARM_LPM_dims_dict
from disdrodb.metadata import read_metadata
from disdrodb.L0_proc import get_file_list

verbose = False
debugging_mode = True
force = True
raw_dir = "/ltenas3/0_Data/DISDRODB/Raw/ARM/SAIL"  #  "data/gucldS2/gucldS2.b1.20210903.100500.cdf"
processed_dir = "/tmp/Processed/ARM/SAIL"

list_stations_id = os.listdir(os.path.join(raw_dir, "data"))
station_id = list_stations_id[1]


attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)
sensor_name = attrs["sensor_name"]

glob_pattern = os.path.join("data", station_id, "*.cdf")

file_list = get_file_list(
    raw_dir=raw_dir,
    glob_pattern=glob_pattern,
    verbose=verbose,
    debugging_mode=debugging_mode,
)

ds = xr.open_mfdataset(file_list)
