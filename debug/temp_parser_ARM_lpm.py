#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:56:11 2022

@author: ghiggi
"""
# ARM LPM Alaska 
import os 
import glob
import xarray as xr

fpath = "/ltenas3/0_Data/DISDRODB/Raw/ARM/ALASKA/data/nsalpmC1/nsalpmC1.a1.20170429.000800.nc"
fpath1 = "/ltenas3/0_Data/DISDRODB/Raw/ARM/ALASKA/data/olilpmM1/olilpmM1.a1.20170724.170000.nc"

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
# Check vars difference 
vars1 = set(ds.data_vars) 
vars2 = set(ds1.data_vars)
vars1.difference(vars2)
vars2.difference(vars1)

# Check bins 
ds.particle_diameter_bounds
ds.particle_fall_velocity_bounds

# Get useful attributes for metadata
ds.attrs["serial_number"]
ds.attrs["sampling_interval"]
ds.attrs["doi"]
ds.lat
ds.lon
ds.alt

 
# FULL CHECK
campaigns = ["ALASKA"]
base_dir = "/ltenas3/0_Data/DISDRODB/Raw/ARM/"

fpath = "/ltenas3/0_Data/DISDRODB/Raw/ARM/ALASKA/data/nsalpmC1/nsalpmC1.a1.20170429.000800.nc"
ds_ref = xr.open_dataset(fpath)
vars_ref = set(ds_ref.data_vars) 

list_stations_pattern = [os.path.join(base_dir, campaign, "data/*") for campaign in campaigns]
list_stations_fpaths = [glob.glob(p) for p in list_stations_pattern]
list_stations_fpaths = [x for xs in list_stations_fpaths for x in xs] # flatten list 

list_station_file_example = [glob.glob(os.path.join(d, "*.nc"))[0] for d in list_stations_fpaths]
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
      
# Look at attributes 
for f in list_station_file_example:
    print(f)
    ds = xr.open_dataset(f)
    # print(ds.attrs["sampling_interval"])   # 1 minute
    # print(ds.attrs["doi"])  # always 10.5439/1390571
    # print(ds.attrs.get("serial_number",'ATTR NOT AVAILABLE')) # 06160016, 06160016, 06160017

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

# Encodings
from disdrodb.metadata import read_metadata
from disdrodb.L0_proc import get_file_list
verbose = False
debugging_mode = True
force = True

raw_dir = "/ltenas3/0_Data/DISDRODB/Raw/ARM/ALASKA"   
processed_dir = "/tmp/Processed/ARM/ALASKA"

list_stations_id = os.listdir(os.path.join(raw_dir, "data"))
station_id = list_stations_id[1]
     
        
attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)
sensor_name = attrs['sensor_name']

glob_pattern= os.path.join("data", station_id, "*.nc")

file_list = get_file_list(
    raw_dir=raw_dir,
    glob_pattern=glob_pattern,
    verbose=verbose,
    debugging_mode=debugging_mode,
)
    
ds = xr.open_mfdataset(file_list)




  
        
 
 
 
 
 

 