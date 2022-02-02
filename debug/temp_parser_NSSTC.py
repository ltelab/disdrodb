#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 09:49:34 2022

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
import pyarrow as pa


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
raw_dir = "/SharedVM/Campagne/GPM/raw/NSSTC"
processed_dir = "/SharedVM/Campagne/GPM/processed/NSSTC"

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

####--------------------------------------------------------------------------.     
##########################################################################   
#### 4. List files to process  [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
##########################################################################
glob_pattern = os.path.join("data", station_id, "*.tar*") # CUSTOMIZE THIS 
device_path = os.path.join(raw_dir, glob_pattern)
file_list = sorted(glob.glob(device_path, recursive = True))
#-------------------------------------------------------------------------. 
# All files into the campaing
all_stations_files = sorted(glob.glob(os.path.join(raw_dir, "data", "*/*.tar"), recursive = True))



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

column_names = ['time',
                'station_name',
                'sensor_status',
                'sensor_temperature',
                'RawData'
                ]

import tarfile

import pandas.errors



def parse(df_temp):
    
    df_temp.iloc[:,0] = pd.to_datetime(df_temp.iloc[:,0], format='%Y%m%d%H%M%S')
    
    df_temp = pd.concat([df_temp.iloc[:,0], df_temp.iloc[:,1].str.split(',', expand=True, n = 3)] ,axis=1)
    
    # Add names to the columns
    df_temp.columns = column_names
    
    return df_temp



df = pd.DataFrame()

for f in all_stations_files[:20]:
    print('')
    print(f'--- {f} ---')
    print('')
    tar = tarfile.open(f)
    print("- files found: %s" % (len(tar.getnames())))
    print('')
    for file in tar.getnames():
        print(f'- {file} -')
        print('')
        try:
            df_temp = pd.read_csv(tar.extractfile(file), compression='infer', header=None, sep=';')
            
            df_temp = parse(df_temp)
            
            df = df.append(df_temp)
            
            
        except pd.errors.EmptyDataError:
            print(f"Is empty, skip file: {file} in {f}")
            pass
        except pd.errors.ParserError:
            print(f"Cannot parse, skip file: {file} in {f}")
            pass
        except UnicodeDecodeError:
            print(f"Unicode error, skip file: {file} in {f}")
            pass
    
    tar.close()






# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




####--------------------------------------------------------------------------. 
#########################################################################
#### 5. Define reader options [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
#########################################################################
# Important: document argument need/behaviour 
    
reader_kwargs = {}

# - Need for zipped raw file (GPM files)
reader_kwargs['zipped'] = True

# - Define delimiter
reader_kwargs['delimiter'] = '   '

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
reader_kwargs['na_values'] = ['na', '', 'error', 'NA', '-.-']

# - Define max size of dask dataframe chunks (if lazy=True)
#   - If None: use a single block for each file
#   - Otherwise: "<max_file_size>MB" by which to cut up larger files
reader_kwargs["blocksize"] = None # "50MB" 

# Cast all to string
reader_kwargs["dtype"] = str

# Skip first row as columns names
reader_kwargs['header'] = None

# Skip file with encoding errors
reader_kwargs['encoding_errors'] = 'ignore'

# Different enconding for this campaign
# reader_kwargs['encoding'] = 'latin-1'  # Important for this campaign

# Use for Nan value
# reader_kwargs['assume_missing'] = True




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@





# ####--------------------------------------------------------------------------. 
# #################################################### 
# #### 6. Open a single file and explore the data ####
# #################################################### 
# # - Do not assign column names yet to the columns 
# # - Do not assign a dtype yet to the columns 
# # - Possibily look at multiple files ;)
# # filepath = file_list[0]
# filepath = file_list[0]
# str_reader_kwargs = reader_kwargs.copy() 
# df = read_raw_data(filepath, 
#                    column_names=None,  
#                    reader_kwargs=str_reader_kwargs, 
#                    lazy=False)

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




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



####---------------------------------------------------------------------------.
######################################################################
#### 7. Define dataframe columns [TO CUSTOMIZE AND MOVE TO PARSER] ###
######################################################################
# - If a column must be splitted in two (i.e. lat_lon), use a name like: TO_SPLIT_lat_lon

column_names = ['time',
                'Unknow',
                'RawData',
                ]



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


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
# # filepath = all_stations_files
# lazy = False             # Try also with True when work with False 

# #------------------------------------------------------. 
# #### 8.1 Run following code portion without modifying anthing 
# # - This portion of code represent what is done by read_L0_raw_file_list in L0_proc.py
# df = read_raw_data(filepath=filepath, 
#                     column_names=columns_names_temporary,
#                     reader_kwargs=reader_kwargs,
#                     lazy=lazy)


# #------------------------------------------------------. 
# # Check if file empty
# # if len(df.index) == 0:
# #     raise ValueError(f"{filepath} is empty and has been skipped.")

# # Check column number
# # if len(df.columns) != len(column_names):
# #     raise ValueError(f"{filepath} has wrong columns number, and has been skipped.")

# #---------------------------------------------------------------------------.  
# #### 8.2 Ad-hoc code [TO CUSTOMIZE]
# # --> Here specify columns to drop, to split and other type of ad-hoc processing     
# # --> This portion of code will need to be enwrapped (in the parser file) 
# #     into a function called df_sanitizer_fun(df, lazy=True). See below ...     
            
# # # Example: split erroneous columns  
# # df_tmp = df['TO_BE_SPLITTED'].astype(str).str.split(',', n=1, expand=True)
# # df_tmp.columns = ['datalogger_voltage','rain_rate_32bit']
# # df = df.drop(columns=['TO_BE_SPLITTED'])
# # df = dd.concat([df, df_tmp], axis = 1, ignore_unknown_divisions=True)
# # del df_tmp 

# # - Convert time column to datetime 
# df['time'] = dd.to_datetime(df['time'], format='%Y%m%d%H%M%S')

# # Split the last column (contain all the fields)
# df_to_parse = df['TO_BE_SPLITTED'].str.split(',', expand=True, n = 1032)

# # Drop TO_BE_SPLITTED
# df = df.drop(['TO_BE_SPLITTED'], axis=1)

# # Add the comma on RawData
# df_RawData = df_to_parse.iloc[:,9:].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1).to_frame('RawData')

# # Concat all togheter
# df = dd.concat([df, df_to_parse.iloc[:,:9], df_RawData] ,axis=1)

# # Add names to the columns
# df.columns = column_names

# # Drop rows with less than 4096 on RawData
# df = df.loc[df['RawData'].astype(str).str.len() == 4096]

# # Drop not float on rain_rate_32bit
# # df = df[pd.to_numeric(df['rain_rate_32bit'], errors='coerce').notnull()]

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
        
# #---------------------------------------------------------------------------.
# #### 8.4 Check the dataframe looks as desired 
# print_df_column_names(df)
# print_df_random_n_rows(df, n= 5) 
# print_df_columns_unique_values(df, column_indices=2, column_names=True) 
# print_df_columns_unique_values(df, column_indices=slice(0,20), column_names=True)   



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# if reader_kwargs.get('zipped'):
#     print('lol')


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

    # - Convert time column to datetime 
    try:
        df['time'] = dd.to_datetime(df['time'], format='%Y %m %d %H %M %S')
    except ValueError:
        df['time'] = dd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    
    return df 

##------------------------------------------------------. 
#### 9.2 Launch code as in the parser file 
# - Try with increasing number of files 
# - Try first with lazy=False, then lazy=True 
lazy = True # True 
subset_file_list = all_stations_files[:]
# subset_file_list = all_stations_files
df = read_L0_raw_file_list(file_list=subset_file_list, 
                           column_names=column_names, 
                           reader_kwargs=reader_kwargs,
                           df_sanitizer_fun = df_sanitizer_fun, 
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
parquet_dir = os.path.join(processed_dir, 'L0', campaign_name + '_s08.parquet')

# Define writing options 
compression = 'snappy' # 'gzip', 'brotli, 'lz4', 'zstd'
row_group_size = 100000 
engine = "pyarrow"

df2 = df.to_parquet(parquet_dir , 
                    # schema = 'infer',
                    engine = engine,
                    row_group_size = row_group_size,
                    compression = compression
                  )
##------------------------------------------------------. 
#### 10.1 Read parquet file
df2 = dd.read_parquet(parquet_dir)
df2 = df2.compute()
print(df2)
