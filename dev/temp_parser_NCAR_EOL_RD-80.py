#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:01:19 2022

@author: kimbo
"""
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

### THIS SCRIPT PROVIDE A columns_names_temporaryLATE FOR PARSER FILE DEVELOPMENT 
#   FROM RAW DATA FILES 
# - Please copy such columns_names_temporarylate and modify it for each parser ;) 
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

from disdrodb.L1_proc import create_L1_dataset_from_L0 
from disdrodb.metadata import read_metadata

##------------------------------------------------------------------------. 
######################################
#### 1. Define campaign filepaths ####
######################################
raw_dir = "/home/kimbo/data/Campagne/DISDRODB/Raw/NCAR/CINDY_2011"
processed_dir = "/home/kimbo/data/Campagne/DISDRODB/Processed/NCAR/CINDY_2011"

l0_processing = True
l1_processing = True
force = True
verbose = True
debugging_mode = True
lazy = True
write_zarr = True
write_netcdf = True

####--------------------------------------------------------------------------.
############################################# 
#### 2. Here run code to not be modified ####
############################################# 
# Initial directory checks 
raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

# Retrieve campaign name 
campaign_name = get_campaign_name(raw_dir)

#-------------------------------------------------------------------------. 
# Define logging settings
# create_logger(processed_dir, 'parser_' + campaign_name) 

# # Retrieve logger 
# logger = logging.getLogger(campaign_name)
# logger.info('### Script start ###')
    
#-------------------------------------------------------------------------. 
# Create directory structure 
create_directory_structure(raw_dir, processed_dir)
               
#-------------------------------------------------------------------------. 
# List stations 
list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

####--------------------------------------------------------------------------. 
###################################################### 
#### 3. Select the station for parser development ####
######################################################
station_id = list_stations_id[0]

attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)
# Retrieve sensor name
sensor_name = attrs['sensor_name']

####--------------------------------------------------------------------------.     
##########################################################################   
#### 4. List files to process  [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
##########################################################################
# glob_pattern = os.path.join("data", station_id, "*.csv*") # CUSTOMIZE THIS 
glob_pattern = os.path.join("data", station_id, "*.txt") # CUSTOMIZE THIS 
device_path = os.path.join(raw_dir, glob_pattern)
file_list = sorted(glob.glob(device_path, recursive = True))
#-------------------------------------------------------------------------. 
# All files into the campaing
# all_stations_files = sorted(glob.glob(os.path.join(raw_dir, "data", "*/*/*/*.csv*"), recursive = True))
# file_list = ['/SharedVM/Campagne/ltnas3/Raw/PAYERNE_2014/data/10/10_ascii_20140324.dat']
# file_list = get_file_list(raw_dir=raw_dir,
#                           glob_pattern=glob_pattern, 
#                           verbose=verbose, 
#                           debugging_mode=debugging_mode)

####--------------------------------------------------------------------------. 
#########################################################################
#### 5. Define reader options [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
#########################################################################
# Important: document argument need/behaviour 
    
reader_kwargs = {}
# - Define delimiter
reader_kwargs['delimiter'] = '\t'

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
# reader_kwargs['compression'] = 'xz'

# - Strings to recognize as NA/NaN and replace with standard NA flags 
#   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, 
#                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’, 
#                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
reader_kwargs['na_values'] = ['na', '', 'error', 'NA', '-.-', ' NA', '-Inf']

# - Define max size of dask dataframe chunks (if lazy=True)
#   - If None: use a single block for each file
#   - Otherwise: "<max_file_size>MB" by which to cut up larger files
reader_kwargs["blocksize"] = None # "50MB" 

# Cast all to string
reader_kwargs["dtype"] = str

# # Skip first row as columns names
reader_kwargs['header'] = None

# # Skip header
reader_kwargs['skiprows'] = 1


####--------------------------------------------------------------------------. 
#################################################### 
#### 6. Open a single file and explore the data ####
#################################################### 
# - Do not assign column names yet to the columns 
# - Do not assign a dtype yet to the columns 
# - Possibily look at multiple files ;)
# filepath = file_list[0]



filepath = file_list[0]

str_reader_kwargs = reader_kwargs.copy() 
df = read_raw_data(filepath, 
                    column_names=None,  
                    reader_kwargs=str_reader_kwargs, 
                    lazy=False).add_prefix('col_').head(n=100)

# df = read_raw_data(filepath, 
#                     column_names=None,  
#                     reader_kwargs=str_reader_kwargs, 
#                     lazy=False)

df['time'] = pd.to_datetime(df['col_0'] + '-' + df['col_1'], format='%Y-%m-%d-%H:%M:%S')

df = df.drop(columns = ['col_0', 'col_1'])

col_names = [
            'n1',
            'n2',
            'n3',
            'n4',
            'n5',
            'n6',
            'n7',
            'n8',
            'n9',
            'n10',
            'n11',
            'n12',
            'n13',
            'n14',
            'n15',
            'n16',
            'n17',
            'n18',
            'n19',
            'n20',
            'DMax', # [mm]
            'R', # [mm/h]
            'RA', # [mm]
            'Wg', # [g/m^3]
            'Z', # [dB]
            'EF', # [J/(m^2 * h)]
            'No', # [1/(m^3 * mm)]
            'Lamda', # [1/mm]
            'time',
            ]

df.columns = col_names

col_raw_drop_number = [
                        'n1',
                        'n2',
                        'n3',
                        'n4',
                        'n5',
                        'n6',
                        'n7',
                        'n8',
                        'n9',
                        'n10',
                        'n11',
                        'n12',
                        'n13',
                        'n14',
                        'n15',
                        'n16',
                        'n17',
                        'n18',
                        'n19',
                        'n20'
                        ]

df[col_raw_drop_number] = df[col_raw_drop_number].astype('str')

df['raw_drop_number'] = df[col_raw_drop_number].T.agg(','.join)

df = df.drop(columns = col_raw_drop_number)

# # Print first rows
# print_df_first_n_rows(df, n = 1, column_names=False)
# print_df_first_n_rows(df, n = 5, column_names=False)
# print_df_random_n_rows(df, n= 5, column_names=False)  # this likely the more useful 

# # Retrieve number of columns 
# print(len(df.columns))
 
# # Look at unique values
# # print_df_columns_unique_values(df, column_indices=None, column_names=False) # all 
 
# # print_df_columns_unique_values(df, column_indices=0, column_names=False) # single column 

# # print_df_columns_unique_values(df, column_indices=slice(0,15), column_names=False) # a slice of columns 

# # get_df_columns_unique_values_dict(df, column_indices=slice(0,15), column_names=False) # get dictionary

# # Retrieve number of columns 
# print(len(df.columns))
 
# # Copy the following list and start to infer column_names 
# ['Unknown' + str(i+1) for i in range(len(df.columns))]

# # Print valid column names 
# # - If other names are required, add the key to get_L0_dtype_standards in data_encodings.py 
# print_valid_L0_column_names()

# # Instrument manufacturer defaults 
# from disdrodb.standards import get_OTT_Parsivel_dict, get_OTT_Parsivel2_dict
# get_OTT_Parsivel_dict()
# get_OTT_Parsivel2_dict()


# ####---------------------------------------------------------------------------.
# ######################################################################
# #### 7. Define dataframe columns [TO CUSTOMIZE AND MOVE TO PARSER] ###
# ######################################################################
# # - If a column must be splitted in two (i.e. lat_lon), use a name like: TO_SPLIT_lat_lon

column_names = ['TO_SPLIT']

# # - Check name validity 
# check_L0_column_names(column_names)

# # - Read data
# # Added function read_raw_data_dtype() on L0_proc for read with columns and all dtypes as object
# filepath = file_list[0]
# df = read_raw_data(filepath=filepath, 
#                     column_names=column_names,
#                     reader_kwargs=reader_kwargs,
#                     lazy=False)
    


# # - Look at the columns and data 
# print_df_column_names(df)
# print_df_random_n_rows(df, n= 5)

# # - Check it loads also lazily in dask correctly
# df1 = read_raw_data(filepath=filepath, 
#                     column_names=column_names,
#                     reader_kwargs=reader_kwargs,
#                     lazy=True)

# df1 = df1.compute() 

# # - Look at the columns and data 
# print_df_column_names(df1)
# print_df_random_n_rows(df1, n= 5) 

# # - Check are equals 
# assert df.equals(df1)

# # - Look at unique values
# print_df_columns_unique_values(df, column_indices=None, column_names=True) # all 
 
# print_df_columns_unique_values(df, column_indices=0, column_names=True) # single column 

# print_df_columns_unique_values(df, column_indices=slice(0,10), column_names=True) # a slice of columns 

# get_df_columns_unique_values_dict(df, column_indices=slice(0,15), column_names=True) # get dictionary

# ####---------------------------------------------------------------------------.
# #########################################################
# #### 8. Implement ad-hoc processing of the dataframe ####
# #########################################################
# # - This must be done once that reader_kwargs and column_names are correctly defined 
# # - Try the following code with various file and with both lazy=True and lazy=False 
# filepath = file_list[0]  # Select also other files here  1,2, ... 
# lazy = False             # Try also with True when work with False 

# #------------------------------------------------------. 
# #### 8.1 Run following code portion without modifying anthing 
# # - This portion of code represent what is done by read_L0_raw_file_list in L0_proc.py
# df = read_raw_data(filepath=filepath, 
#                     column_names=column_names,
#                     reader_kwargs=reader_kwargs,
#                     lazy=lazy).head(n=100)

# #------------------------------------------------------. 
# # Check if file empty
# # if len(df.index) == 0:
# #     raise ValueError(f"{filepath} is empty and has been skipped.")

# # # Check column number
# # if len(df.columns) != len(column_names):
# #     raise ValueError(f"{filepath} has wrong columns number, and has been skipped.")

# #---------------------------------------------------------------------------.  
# #### 8.2 Ad-hoc code [TO CUSTOMIZE]
# # --> Here specify columns to drop, to split and other type of ad-hoc processing     
# # --> This portion of code will need to be enwrapped (in the parser file) 
# #     into a function called df_sanitizer_fun(df, lazy=True). See below ...     
            
# # # Example: split erroneous columns  
# # df_tmp = df['TO_BE_SPLITTED'].astype(str).str.split(',', n=1, expand=True)
# # df_tmp.columns = ['datalogger_voltage','rainfall_rate_32bit']
# # df = df.drop(columns=['TO_BE_SPLITTED'])
# # df = dd.concat([df, df_tmp], axis = 1, ignore_unknown_divisions=True)
# # del df_tmp 

# # Add prefix to columns
# # df = df.add_prefix('col_')

# # ----

# # Read date from header
# date = str(df.iloc[0])
# date = date[:10]

# # Define temporary column
# df.columns = ['TO_SPLIT']

# # Split time
# df[['time','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)
# df['time'] = df['time'].str[:-3]
# df['time'] = df['time'] + '-' + date
# df['time'] = pd.to_datetime(df['time'], format='%M%S-%d/%m/%Y')

# # Remove useless spaces
# df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

# df[['rainfall_rate_32bit','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

# df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

# df[['rainfall_accumulated_32bit','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

# df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

# df[['reflectivity_32bit','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

# df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

# df[['number_particles','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

# df[['sensor_status','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

# df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

# df[['error_code','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

# df[['raw_drop_concentration','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

# df[['raw_drop_average_velocity','raw_drop_number']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

# df = df.drop(columns = ['TO_SPLIT'])

# # Add 0 digits to raw_drop_number
# for i, r in df.iterrows():
#     raw = ''
#     for n in r['raw_drop_number']:
#         if n != ',':
#             n = '%03d' % int(n) + ','
#             raw += n
#     df['raw_drop_number'] = raw

# #---------------------------------------------------------------------------.
# #### 8.3 Run following code portion without modifying anthing 
# # - This portion of code represent what is done by read_L0_raw_file_list in L0_proc.py

# # ## Keep only clean data 
# # # - This type of filtering will be done in the background automatically ;) 
# # # Remove rows with bad data 
# # # df = df[df.sensor_status == 0] 
# # # Remove rows with error_code not 000 
# # # df = df[df.error_code == 0]  

# ##----------------------------------------------------.
# # Cast dataframe to dtypes
# # - Determine dtype based on standards 
# dtype_dict = get_L0_dtype_standards()
# for column in df.columns:
#     try:
#         df[column] = df[column].astype(dtype_dict[column])
#     except KeyError:
#         # If column dtype is not into get_L0_dtype_standards, assign object
#         df[column] = df[column].astype('object')
#     except ValueError as e:
#         print(f"The column {column} has {e}")
        
# #---------------------------------------------------------------------------.
# #### 8.4 Check the dataframe looks as desired 
# print_df_column_names(df)
# print_df_random_n_rows(df, n= 5) 
# print_df_columns_unique_values(df, column_indices=2, column_names=True) 
# print_df_columns_unique_values(df, column_indices=slice(0,20), column_names=True)   

####------------------------------------------------------------------------------.
################################################
#### 9. Simulate parser file code execution #### 
################################################
#### 9.1 Define sanitizer function [TO CUSTOMIZE]
# --> df_sanitizer_fun = None  if not necessary ...

def df_sanitizer_fun(df, lazy=False):
    # Import dask or pandas 
    if lazy: 
        import dask.dataframe as dd
    else: 
        import pandas as dd
    
    # Read date from header and then remove it
    if lazy:
        temp = df.loc[0].compute()
        date = temp.iloc[0][0][:10]
    else:
        date = df.iloc[0][0][:10]
    df = df.loc[1:]

    # Split time
    df[['time','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)
    df['time'] = df['time'].str[:-3]
    df['time'] = df['time'] + '-' + date
    df['time'] = dd.to_datetime(df['time'], format='%M%S-%m/%d/%Y')

    # Remove useless spaces
    df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

    df[['rainfall_rate_32bit','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

    df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

    df[['rainfall_accumulated_32bit','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

    df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

    df[['reflectivity_32bit','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

    df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

    df[['number_particles','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

    df[['sensor_status','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

    df['TO_SPLIT'] = df['TO_SPLIT'].str.strip()

    df[['error_code','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

    df[['raw_drop_concentration','TO_SPLIT']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

    df[['raw_drop_average_velocity','raw_drop_number']] = df['TO_SPLIT'].str.split(' ', n=1, expand=True)

    df = df.drop(columns = ['TO_SPLIT'])
    
    # Chech for integrity data
    df = df.loc[df['raw_drop_number'].str.count(',') == 1023]

    # Add 0 digits to raw_drop_number
    for i, r in df.iterrows():
        raw = ''
        t = r['raw_drop_number'].split(',')
        
        for n in t:
            n = '%03d' % int(n) + ','
            raw += n
        df['raw_drop_number'] = raw
        
    # Delete index
    # df = df.set_index('time')
        
    return df 



##------------------------------------------------------. 
#### 9.2 Launch code as in the parser file 
# - Try with increasing number of files 
# - Try first with lazy=False, then lazy=True
lazy = False # True 
subset_file_list = file_list[:3]
# subset_file_list = all_stations_files
df = read_L0_raw_file_list(file_list=subset_file_list, 
                           column_names=column_names, 
                           reader_kwargs=reader_kwargs,
                           df_sanitizer_fun = df_sanitizer_fun, 
                           sensor_name = sensor_name,
                           verbose = verbose,
                           lazy=lazy)

##------------------------------------------------------. 
#### 9.3 Check everything looks goods
df = df.compute() # if lazy = True 
print_df_column_names(df)
print_df_random_n_rows(df, n= 5) 
print_df_columns_unique_values(df, column_indices=2, column_names=True) 
print_df_columns_unique_values(df, column_indices=slice(0,20), column_names=True)  


infer_df_str_column_names(df, 'Parsivel')

####--------------------------------------------------------------------------. 
##------------------------------------------------------. 
#### 10. Conversion to parquet
parquet_dir = os.path.join(processed_dir, 'L0', campaign_name + '_s10.parquet')

# Define writing options 
compression = 'snappy' # 'gzip', 'brotli, 'lz4', 'zstd'
row_group_size = 100000 
engine = "pyarrow"

df_to_parse = df.to_parquet(parquet_dir , 
                    # schema = 'infer',
                    engine = engine,
                    row_group_size = row_group_size,
                    compression = compression
                  )
##------------------------------------------------------. 
#### 10.1 Read parquet file
df_to_parse = dd.read_parquet(parquet_dir)
df_to_parse = df_to_parse.compute()
print(df_to_parse)


####--------------------------------------------------------------------------. 
##------------------------------------------------------. 
#### 20. Process L1

#-----------------------------------------------------------------.
#### 20.1 Create xarray Dataset
attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)
# Retrieve sensor name
sensor_name = attrs['sensor_name']

ds = create_L1_dataset_from_L0(df=df, attrs=attrs, lazy=lazy, verbose=verbose)