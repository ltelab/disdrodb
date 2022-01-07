#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:47:24 2022

@author: ghiggi
"""
import os
os.chdir("/ltenas3/0_Projects/disdrodb")

import os
import click
import time
import logging
import xarray as xr 

# Directory 
from disdrodb.io import check_directories
from disdrodb.io import get_campaign_name
from disdrodb.io import create_directory_structure

# Metadata 
from disdrodb.metadata import read_metadata

# IO 
from disdrodb.io import get_L0_fpath
from disdrodb.io import get_L1_netcdf_fpath
from disdrodb.io import get_L1_zarr_fpath  
from disdrodb.io import read_L0_data

# L0_processing
from disdrodb.check_standards import check_L0_column_names 
from disdrodb.check_standards import check_L0_standards
from disdrodb.L0_proc import get_file_list
from disdrodb.L0_proc import read_L0_raw_file_list
from disdrodb.L0_proc import write_df_to_parquet

# L1_processing
from disdrodb.L1_proc import create_L1_dataset_from_L0 
from disdrodb.L1_proc import write_L1_to_zarr
from disdrodb.L1_proc import write_L1_to_netcdf
from disdrodb.L1_proc import create_L1_summary_statistics
   
# Logger 
from disdrodb.logger import create_logger
from disdrodb.logger import close_logger

l0_processing = True
l1_processing = True
force = True
verbose = True
debugging_mode = True
lazy = False
write_zarr = True
write_netcdf = True

raw_dir = "/ltenas3/0_Data/ParsivelDB/Raw/EPFL/EPFL_ROOF_2011"
processed_dir = "/ltenas3/0_Data/ParsivelDB/Processed/EPFL/EPFL_ROOF_2011"

raw_dir = "/ltenas3/0_Data/ParsivelDB/Raw/EPFL/DAVOS_2009_2011"
processed_dir = "/ltenas3/0_Data/ParsivelDB/Processed/EPFL/DAVOS_2009_2011"

campaign_name = get_campaign_name(raw_dir)

list_stations_id = os.listdir(os.path.join(raw_dir, "data"))
station_id = list_stations_id[1]  
station_id = list_stations_id[0]  

# Retrieve metadata 
attrs = read_metadata(raw_dir=raw_dir,
                      station_id=station_id)
# Retrieve sensor name
sensor_name = attrs['sensor_name']
        
lazy = False 
debugging_mode  = False 
df = read_L0_data(processed_dir, station_id, lazy=lazy, verbose=verbose, debugging_mode=debugging_mode)
print(df.shape)
print(df.columns)

import numpy as np 
from disdrodb.L1_proc import retrieve_L1_raw_data_matrix 
from disdrodb.L1_proc import create_L1_dataset_from_L0 

np.unique(df['RawData'].astype(str).str.len())
np.unique(df['FieldN'].astype(str).str.len())  # CHECK TO ADD MAYBE TO DF_SANITIZER
np.unique(df['FieldV'].astype(str).str.len())

lazy = False
df = read_L0_data(processed_dir, station_id, lazy=lazy, verbose=verbose, debugging_mode=debugging_mode)
df_sub = df.iloc[0:100000,:]
dict_data = retrieve_L1_raw_data_matrix(df_sub, sensor_name, lazy=lazy, verbose=verbose)
ds = create_L1_dataset_from_L0(df=df_sub, attrs=attrs, lazy=lazy, verbose=verbose)

lazy = True
df = read_L0_data(processed_dir, station_id, lazy=lazy, verbose=verbose, debugging_mode=debugging_mode)
dict_data = retrieve_L1_raw_data_matrix(df_sub, sensor_name, lazy=lazy, verbose=verbose)
ds = create_L1_dataset_from_L0(df=df_sub, attrs=attrs, lazy=lazy, verbose=verbose)
ds.compute()

df_series = df_sub["RawData"].astype(str).str.split(",").iloc[0]

df_series.shape
type(df_series)
len(df_series[0].values[1])

variable = "RawData"
variable = "sensor_heating_current"
df_sub.iloc[0,:][variable]
df_sub[variable] # series 
df_sub[[variable]] # df 
df_sub[[variable]].iloc[0]

np.char.split(df_sub[variable].astype(str).iloc[0], ",")


df_sub["RawData"][0].astype(str).str.split(",")[0][0]
 
v = df_sub["RawData"][0].values.astype(str)
a = np.char.split(v, ",")
len(a[0])
len(a[1])
np.allclose(np.array(a[0]).astype(int), np.array(a[1]).astype(int))


# Remove '' at the last array position  
arr = arr[: , 0:n_bins_dict[key]]
