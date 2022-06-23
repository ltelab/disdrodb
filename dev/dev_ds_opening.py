#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:13:48 2022

@author: ghiggi
"""
import time 
import os 
from disdrodb.metadata import read_metadata
from disdrodb.check_standards import check_sensor_name
from disdrodb.io import get_campaign_name
# L0 processing
from disdrodb.L0_proc import get_file_list

#### Define filepaths
campaign_dict = {
    "ACE_ENA": "parser_ARM_ld.py",
    "AWARE": "parser_ARM_ld.py",
    "CACTI": "parser_ARM_ld.py",
    "COMBLE": "parser_ARM_ld.py",
    "GOAMAZON": "parser_ARM_ld.py",
    "MARCUS": "parser_ARM_ld.py", # MARCUS S1, MARCUS S2 are mobile ...
    "MICRE": "parser_ARM_ld.py",
    "MOSAIC": "parser_ARM_ld.py", # MOSAIC M1, MOSAIC S3 are mobile ...
    "SAIL": "parser_ARM_ld.py",
    "SGP": "parser_ARM_ld.py",
    "TRACER": "parser_ARM_ld.py",
    "ALASKA": "parser_ARM_lpm.py",
}     
parser_dir = "/ltenas3/0_Projects/disdrodb/disdrodb/readers/ARM" # TO CHANGE
raw_base_dir = "/ltenas3/0_Data/DISDRODB/Raw/ARM"
processed_base_dir = "/ltenas3/0_Data/DISDRODB/Processed/ARM"
# processed_base_dir = "/tmp/DISDRODB/ARM"

#### Processing settings
campaign_name = "ACE_ENA"
force = True
verbose = True
debugging_mode = False
lazy = True

#### Process all campaigns

print("Processing: ", campaign_name)
parser_filepath = os.path.join(parser_dir, campaign_dict[campaign_name])
raw_dir = os.path.join(raw_base_dir, campaign_name)
processed_dir = os.path.join(processed_base_dir, campaign_name)

raw_data_glob_pattern = "*.cdf"
# Retrieve campaign name
campaign_name = get_campaign_name(raw_dir)
# Get station list 
list_stations_id = os.listdir(os.path.join(raw_dir, "data"))
station_id = list_stations_id[0]
# Retrieve metadata
attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)
# Retrieve sensor name
sensor_name = attrs['sensor_name']
check_sensor_name(sensor_name)
# Retrieve list of files to process
glob_pattern = os.path.join("data", station_id, raw_data_glob_pattern)
file_list = get_file_list(
    raw_dir=raw_dir,
    glob_pattern=glob_pattern,
    verbose=verbose,
    debugging_mode=debugging_mode,
)
        
#### Open netCDFs
import dask 
from dask.distributed import Client
from disdrodb.L0.utils_nc import xr_concat_datasets

client = Client(processes=True) # n_workers=2, threads_per_worker=2
client.ncores()
client.nthreads()

t_i = time.time() 
ds = xr_concat_datasets(file_list)
t_f = time.time()
print(t_f - t_i) 

# xr.openmdataset
# - combine_by_coords use xr.concat
# - combine_nested use xr.merge 
# - combine_nested manage to work with overlapping coordinates 
# - combine_by_coords can not handle overlapping coordinates

# xr.concat 
# - With the default parameters, xarray will load some coordinate variables into memory to compare them between datasets.
# - Default parameters may be prohibitively expensive if you are manipulating your dataset lazily
# -- data_vars:  if data variable with no dimension changing over files ... need to specify "all"
# - The concatenate step is never supposed to handle partially overlapping coordinates

  


    
 





