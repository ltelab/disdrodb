#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:56:11 2022

@author: ghiggi
"""
# ------------------------------------------------------------------------------.
# ARM OTT Parsivel
import os
import glob
import xarray as xr

fpath = "/ltenas3/0_Data/DISDRODB/Raw/ARM/ACE_ENA/data/enaldC1/enaldC1.b1.20140227.212600.cdf"
fpath1 = (
    "/ltenas3/0_Data/DISDRODB/Raw/ARM/AWARE/data/awrldM1/awrldM1.b1.20151119.023600.cdf"
)
fpath = (
    "/ltenas3/0_Data/DISDRODB/Raw/ARM/CACTI/data/corldM1/corldM1.b1.20180923.193000.cdf"
)
ds = xr.open_dataset(fpath)
ds1 = xr.open_dataset(fpath1)

ds
ds1

ds.data_vars
ds1.data_vars

ds.coords
ds1.coords

ds.attrs
ds1.attrs

vars1 = set(ds.data_vars)
vars2 = set(ds1.data_vars)
vars1.difference(vars2)
vars2.difference(vars1)

# Check bins
ds.particle_size  # ARM last bin center: 24 instead of 24.5
ds.raw_fall_velocity

# Get useful attributes for metadata
ds.lat
ds.lon
ds.alt
ds.attrs["serial_number"]
ds.attrs["sampling_interval"]
ds.attrs["effective_measurement_area"]

# FULL CHECK
campaigns = [
    "ACE_ENA",
    "AWARE",
    "CACTI",
    "COMBLE",
    "GOAMAZON",
    "MARCUS",
    "MICRE",
    "MOSAIC",
    "SAIL",
    "SGP",
    "TRACER",
]
base_dir = "/ltenas3/0_Data/DISDRODB/Raw/ARM/"

fpath = "/ltenas3/0_Data/DISDRODB/Raw/ARM/ACE_ENA/data/enaldC1/enaldC1.b1.20140227.212600.cdf"
ds_ref = xr.open_dataset(fpath)
vars_ref = set(ds_ref.data_vars)

list_stations_pattern = [
    os.path.join(base_dir, campaign, "data/*") for campaign in campaigns
]
list_stations_fpaths = [glob.glob(p) for p in list_stations_pattern]
list_stations_fpaths = [x for xs in list_stations_fpaths for x in xs]  # flatten list

list_station_file_example = [
    glob.glob(os.path.join(d, "*.cdf"))[0] for d in list_stations_fpaths
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

# COMBLE, MOSAIC, SAIL, TRACER
# --> snow_depth_intensity
ds_tmp = xr.open_dataset(
    "/ltenas3/0_Data/DISDRODB/Raw/ARM/SAIL/data/gucldS2/gucldS2.b1.20210903.100500.cdf"
)
ds_tmp.snow_depth_intensity.attrs["comment"]

# Look at attributes
for f in list_station_file_example:
    print(f)
    ds = xr.open_dataset(f)
    # print(ds.attrs["sampling_interval"])          # 1 minute
    # print(ds.attrs["effective_measurement_area"]) # 0.54 m^2
    # print(ds.attrs.get("serial_number",'ATTR NOT AVAILABLE'))

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


# ---------------------------------------------------------------------------------------
#### DEVELOPMENT
from disdrodb.metadata import read_metadata
from disdrodb.L0_proc import get_file_list

verbose = False
debugging_mode = True
force = True
raw_dir = "/ltenas3/0_Data/DISDRODB/Raw/ARM/SAIL"  #  "data/gucldS2/gucldS2.b1.20210903.100500.cdf"
processed_dir = "/tmp/Processed/ARM/SAIL"

list_stations_id = os.listdir(os.path.join(raw_dir, "data"))
station_id = list_stations_id[0]

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
ds = reformat_ARM_files(file_list, attrs)

ds.sensor_temperature


# ---------------------------------------------------------------------------------------
### CHECK open_mfdataset
campaigns = [
    "ACE_ENA",
    "AWARE",
    "CACTI",
    "COMBLE",
    "GOAMAZON",
    "MARCUS",
    "MICRE",
    "MOSAIC",
    "SAIL",
    "SGP",
    "TRACER",
]
base_dir = "/ltenas3/0_Data/DISDRODB/Raw/ARM/"

for campaign in campaigns:
    raw_dir = os.path.join(base_dir, campaign)
    list_stations_id = os.listdir(os.path.join(raw_dir, "data"))
    for station_id in list_stations_id:
        print(campaign + "-" + station_id)
        glob_pattern = os.path.join("data", station_id, "*.cdf")
        file_list = get_file_list(
            raw_dir=raw_dir,
            glob_pattern=glob_pattern,
            verbose=False,
            debugging_mode=False,
        )
        file_list = sorted(file_list)
        print(len(file_list))

    # import dask
    # with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    # with dask.config.set(**{'array.slicing.split_large_chunks': False}):

    t_i = time.time()
    ds = xr.open_mfdataset(
        file_list[0:100],
        # parallel=True,
        parallel=False,
        combine="nested",
        compat="override",
        #  chunks="auto",
    )
    t_f = time.time()
    print(t_f - t_i)  # parallel = True   # 50 s
    # parallel = False  #  39 s
