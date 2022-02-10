#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:57:33 2022

@author: kimbo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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
#-----------------------------------------------------------------------------.

### THIS SCRIPT PROVIDE A TEMPLATE FOR PARSER FILE DEVELOPMENT 
#   FROM RAW DATA FILES 
# - Please copy such template and modify it for each parser ;) 
# - Additional functions/tools to ease parser development are welcome 

#-----------------------------------------------------------------------------.
import os
import logging
import glob 
import shutil
import pandas as pd 
import dask.dataframe as dd
import dask.array as da
import numpy as np 
import xarray as xr


from disdrodb.io import check_directories
from disdrodb.io import get_campaign_name
from disdrodb.io import create_directory_structure
from disdrodb.check_standards import check_L0_column_names 
from disdrodb.data_encodings import get_L0_dtype_standards
from disdrodb.L0_proc import read_raw_data
from disdrodb.L0_proc import get_file_list
from disdrodb.L0_proc import read_L0_raw_file_list
from disdrodb.L0_proc import write_df_to_parquet
from disdrodb.logger import create_logger

# DEV TOOOLS 
from disdrodb.dev_tools import print_df_first_n_rows 
from disdrodb.dev_tools import print_df_random_n_rows
from disdrodb.dev_tools import print_df_column_names 
from disdrodb.dev_tools import print_valid_L0_column_names
from disdrodb.dev_tools import get_df_columns_unique_values_dict
from disdrodb.dev_tools import print_df_columns_unique_values
from disdrodb.dev_tools import infer_df_str_column_names

##------------------------------------------------------------------------. 
######################################
#### 1. Define campaign filepaths ####
######################################
raw_dir = "/SharedVM/Campagne/EPFL/Raw/PAYERNE_2014"
processed_dir = "/SharedVM/Campagne/EPFL/Processed/PAYERNE_2014"

l0_processing = True
l1_processing = True
force = True
verbose = True
debugging_mode = True
lazy = True
write_zarr = True
write_netcdf = True

# Initial directory checks 
raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

# Retrieve campaign name 
campaign_name = get_campaign_name(raw_dir)

    
#-------------------------------------------------------------------------. 
# Create directory structure 
create_directory_structure(raw_dir, processed_dir)
               
#-------------------------------------------------------------------------. 
# List stations 
list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

station_id = list_stations_id[0]

glob_pattern = os.path.join("data", station_id, "*.dat*") # CUSTOMIZE THIS 
device_path = os.path.join(raw_dir, glob_pattern)
file_list = sorted(glob.glob(device_path, recursive = True))

    
reader_kwargs = {}
# - Define delimiter
reader_kwargs['delimiter'] = ','

# - Avoid first column to become df index !!!
reader_kwargs["index_col"] = False  

# - Define behaviour when encountering bad lines 
reader_kwargs["on_bad_lines"] = 'skip'

# - Define parser engine 
#   - C engine is faster
#   - Python engine is more feature-complete
reader_kwargs["engine"] = 'python'

# - Define on-the-fly decompression of on-disk data
#   - Available: gzip, bz2, zip 
reader_kwargs['compression'] = 'infer'  

# - Strings to recognize as NA/NaN and replace with standard NA flags 
#   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, 
#                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’, 
#                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
reader_kwargs['na_values'] = ['na', '', 'error', 'NA']

# - Define max size of dask dataframe chunks (if lazy=True)
#   - If None: use a single block for each file
#   - Otherwise: "<max_file_size>MB" by which to cut up larger files
reader_kwargs["blocksize"] = None # "50MB" 

# Cast all to string
reader_kwargs["dtype"] = str


column_names = ['time',
                    'id',
                    'datalogger_temperature',
                    'datalogger_voltage',
                    'rain_accumulated_32bit',
                    'Unknow',
                    'weather_code_SYNOP_4680',
                    'weather_code_SYNOP_4677',
                    'reflectivity_32bit',
                    'mor_visibility',
                    'laser_amplitude',
                    'n_particles',
                    'sensor_temperature',
                    'sensor_heating_current',
                    'sensor_battery_voltage',
                    'sensor_status',
                    'rain_amount_absolute_32bit',
                    'Debug_data',
                    'FieldN',
                    'FieldV',
                    'RawData',
                    'All_0'
                    ]



def df_sanitizer_fun(df, lazy=False):
    # # Import dask or pandas 
    # if lazy: 
    #     import dask.dataframe as dd
    # else: 
    #     import pandas as dd

    # - Drop useless columns 
    col_to_drop_if_na = ['FieldN','FieldV','RawData']
    df = df.dropna(subset = col_to_drop_if_na)
 
    # Drop rows with less than 4096 char on RawData
    df = df.loc[df['RawData'].astype(str).str.len() == 4096]
 
    # Example: drop unrequired columns for L0 
    df = df.drop(columns = ['All_0', 'Debug_data'])
    
    # - Convert time column to datetime 
    df['time'] = dd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    
    return df 

##------------------------------------------------------. 
#### 9.2 Launch code as in the parser file 
# - Try with increasing number of files 
# - Try first with lazy=False, then lazy=True 
lazy = True # True 
subset_file_list = file_list[:1]
df = read_L0_raw_file_list(file_list=subset_file_list, 
                           column_names=column_names, 
                           reader_kwargs=reader_kwargs,
                           df_sanitizer_fun = df_sanitizer_fun, 
                           lazy=lazy)

df = df.compute() 






