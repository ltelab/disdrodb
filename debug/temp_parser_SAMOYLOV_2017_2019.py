#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 04:13:26 2022

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

# L1 processing
from disdrodb.L1_proc import create_L1_dataset_from_L0 
from disdrodb.metadata import read_metadata

##------------------------------------------------------------------------. 
######################################
#### 1. Define campaign filepaths ####
######################################
raw_dir = "/SharedVM/Campagne/EPFL/Raw/SAMOYLOV_2017_2019"
processed_dir = "/SharedVM/Campagne/EPFL/Processed/SAMOYLOV_2017_2019"

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
create_logger(processed_dir, 'parser_' + campaign_name) 

# Retrieve logger 
logger = logging.getLogger(campaign_name)
logger.info('### Script start ###')
    
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
glob_pattern = os.path.join("data", station_id, "*.log*") # CUSTOMIZE THIS 
device_path = os.path.join(raw_dir, glob_pattern)
file_list = sorted(glob.glob(device_path, recursive = True))
#-------------------------------------------------------------------------. 
# All files into the campaing
all_stations_files = sorted(glob.glob(os.path.join(raw_dir, "data", "*/*.log*"), recursive = True))
# file_list = ['/SharedVM/Campagne/ltnas3/Raw/PAYERNE_2014/data/10/10_ascii_20140324.dat']

####--------------------------------------------------------------------------. 
#########################################################################
#### 5. Define reader options [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
#########################################################################
# Important: document argument need/behaviour 
    
reader_kwargs = {}
# - Define delimiter
reader_kwargs['delimiter'] = ';'

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
reader_kwargs['na_values'] = ['na', '', 'error', 'NA', 'Error in data reading! 0000.000']

# - Define max size of dask dataframe chunks (if lazy=True)
#   - If None: use a single block for each file
#   - Otherwise: "<max_file_size>MB" by which to cut up larger files
reader_kwargs["blocksize"] = None # "50MB" 

# Cast all to string
reader_kwargs["dtype"] = str

# Different enconding for this campaign
reader_kwargs['encoding'] = 'latin-1'

# Skip first row as columns names
reader_kwargs['header'] = None


# reader_kwargs['date_parser'] = lambda x: pd.to_datetime(x, errors="coerce")


# Retrieve metadata 
attrs = read_metadata(raw_dir=raw_dir,
                      station_id=station_id)
# Retrieve sensor name
sensor_name = attrs['sensor_name']

####--------------------------------------------------------------------------. 
#################################################### 
#### 6. Open a single file and explore the data ####
#################################################### 
# - Do not assign column names yet to the columns 
# - Do not assign a dtype yet to the columns 
# - Possibily look at multiple files ;)
# # filepath = file_list[0]
# filepath = file_list[0]
# str_reader_kwargs = reader_kwargs.copy() 
# df = read_raw_data(filepath, 
#                    column_names=None,  
#                    reader_kwargs=str_reader_kwargs, 
#                    lazy=False).add_prefix('col_')

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


####---------------------------------------------------------------------------.
######################################################################
#### 7. Define dataframe columns [TO CUSTOMIZE AND MOVE TO PARSER] ###
######################################################################
# - If a column must be splitted in two (i.e. lat_lon), use a name like: TO_SPLIT_lat_lon


column_names = ['time',
                'latitude',
                'longitude',
                'weather_code_synop_4680',
                'weather_code_synop_4677',
                'reflectivity_32bit',
                'mor_visibility',
                'laser_amplitude',
                'number_particles',
                'sensor_temperature',
                'sensor_heating_current',
                'sensor_battery_voltage',
                'datalogger_error',
                'rainfall_amount_absolute_32bit',
                'All_0',
                'raw_drop_concentration',
                'raw_drop_average_velocity',
                'raw_drop_number',
                ]

column_names_2 =   ['id',
                    'latitude',
                    'longitude',
                    'time',
                    'all_nan',
                    'rainfall_rate_32bit',
                    'rainfall_accumulated_32bit',
                    'weather_code_synop_4680',
                    'weather_code_synop_4677',
                    'reflectivity_32bit',
                    'mor_visibility',
                    'laser_amplitude',
                    'number_particles',
                    'sensor_temperature',
                    'sensor_heating_current',
                    'sensor_battery_voltage',
                    'All_0',
                    'rainfall_amount_absolute_32bit',
                    'datalogger_error',
                    'raw_drop_concentration',
                    'raw_drop_average_velocity',
                    'raw_drop_number',
                    'End_line'
                    ]

temp_column_names =  ['temp1',
                    'temp2',
                    'temp3',
                    'temp4',
                    'temp5',
                    'temp6',
                    'temp7',
                    'temp8',
                    'temp9',
                    'temp10',
                    'temp11',
                    'temp12',
                    'temp13',
                    'temp14',
                    'temp15',
                    'temp16',
                    'temp17',
                    'temp18',
                    'temp19',
                    'temp20',
                    'temp21',
                    'temp22',
                    'temp23']


# # - Check name validity 
# check_L0_column_names(column_names)

# # - Read data
# # Added function read_raw_data_dtype() on L0_proc for read with columns and all dtypes as object
# filepath = file_list[1]
# filepath = all_stations_files[13]
# filepath = all_stations_files[40]

# df = pd.read_csv('/SharedVM/Campagne/EPFL/Raw/SAMOYLOV_2017_2019/data/01/03-08-2017.PARSIVEL01.log', delim_whitespace=True)
# df = df.iloc[:,1].str.split(';',expand=True, n = 17)

# df = pd.read_csv('/SharedVM/Campagne/EPFL/Raw/SAMOYLOV_2017_2019/data/02/01-08-2017.PARSIVEL2.log', delim_whitespace=True)
# df = df.iloc[:,1].str.split(';',expand=True, n = 17)

# df = pd.read_csv('/SharedVM/Campagne/EPFL/Raw/SAMOYLOV_2017_2019/data/02/01-08-2017.PARSIVEL2.log', delimiter=';')
# df2 = pd.read_csv('/SharedVM/Campagne/EPFL/Raw/SAMOYLOV_2017_2019/data/20/02-09-2019.parsivel20.log', delimiter=';')

# df = read_raw_data(filepath=filepath, 
#                     column_names=column_names_temp_2,
#                     reader_kwargs=reader_kwargs,
#                    lazy=False)

# # - If first column is ID, than is a different format
# if df.iloc[:,0].str.isnumeric().all():
#     # - Rename columns
#     df.columns = column_names_2
#     # - Remove ok from rainfall_rate_32bit
#     df['rainfall_rate_32bit'] = df['rainfall_rate_32bit'].str.split(',').str[-1]
#     # - Drop useless columns
#     col_to_drop = ["id", "all_nan", "All_0", 'datalogger_error', 'End_line']
#     df = df.drop(columns=col_to_drop)
#     # - Convert time column to datetime 
#     df['time'] = dd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
# else:
#     # - Drop excedeed columns
#     df = df.iloc[:,:18]
#     # - Rename columns
#     df.columns = column_names
#     # - Drop useless columns
#     col_to_drop = ["All_0", 'datalogger_error']
#     df = df.drop(columns=col_to_drop)
#     # - Convert time column to datetime 
#     df['time'] = dd.to_datetime(df['time'], format='%d/%m/%Y %H:%M:%S')

# # - Drop columns if nan
# col_to_drop_if_na = ['latitude','longitude','raw_drop_concentration','raw_drop_average_velocity','raw_drop_number']
# df = df.dropna(subset = col_to_drop_if_na)



# # ---------------

# df = df.iloc[:,0].str.split(';',expand=True, n = 22)
# df.columns = column_names_2
# df = df.dropna(subset = ['End_line'])
# df = df.iloc[:,:-1]
    
# df['rainfall_rate_32bit'] = df['rainfall_rate_32bit'].str.split(',').str[-1]

# col_to_drop = ["id", "all_nan", "All_0", 'datalogger_error', 'End_line']

# df = df.drop(columns=col_to_drop)

# col_to_drop_if_na = ['latitude','longitude','raw_drop_concentration','raw_drop_average_velocity','raw_drop_number']
# df = df.dropna(subset = col_to_drop_if_na)

# # - Look at the columns and data 
# print_df_column_names(df)
# print_df_random_n_rows(df, n= 5)

# # - Check it loads also lazily in dask correctly
# df1 = read_raw_data(filepath=filepath, 
#                    column_names=column_names,
#                    reader_kwargs=reader_kwargs,
#                    lazy=True)

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

####---------------------------------------------------------------------------.
#########################################################
#### 8. Implement ad-hoc processing of the dataframe ####
# #########################################################
# # - This must be done once that reader_kwargs and column_names are correctly defined 
# # - Try the following code with various file and with both lazy=True and lazy=False 
# filepath = file_list[0]  # Select also other files here  1,2, ... 
# lazy = True             # Try also with True when work with False 

# #------------------------------------------------------. 
# #### 8.1 Run following code portion without modifying anthing 
# # - This portion of code represent what is done by read_L0_raw_file_list in L0_proc.py
# df = read_raw_data(filepath=filepath, 
#                    column_names=column_names,
#                    reader_kwargs=reader_kwargs,
#                    lazy=lazy)

#------------------------------------------------------. 
# # Check if file empty
# if len(df.index) == 0:
#     raise ValueError(f"{filepath} is empty and has been skipped.")

# # Check column number
# if len(df.columns) != len(column_names):
#     raise ValueError(f"{filepath} has wrong columns number, and has been skipped.")

#---------------------------------------------------------------------------.  
#### 8.2 Ad-hoc code [TO CUSTOMIZE]
# --> Here specify columns to drop, to split and other type of ad-hoc processing     
# --> This portion of code will need to be enwrapped (in the parser file) 
#     into a function called df_sanitizer_fun(df, lazy=True). See below ...     
            
# # Example: split erroneous columns  
# df_tmp = df['TO_BE_SPLITTED'].astype(str).str.split(',', n=1, expand=True)
# df_tmp.columns = ['datalogger_voltage','rainfall_rate_32bit']
# df = df.drop(columns=['TO_BE_SPLITTED'])
# df = dd.concat([df, df_tmp], axis = 1, ignore_unknown_divisions=True)
# del df_tmp 

# # Drop Debug_data and All_0
# df = df.drop(columns=['datalogger_error', 'End_line'])

# df = df[df['latitude'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]

# # If raw_drop_number is nan, drop the row
# # col_to_drop_if_na = ['raw_drop_concentration','raw_drop_average_velocity','raw_drop_number']
# col_to_drop_if_na = ['raw_drop_number']
# df = df.dropna(subset = col_to_drop_if_na)

# # Drop rows with less than 4096 char on raw_drop_number
# # df = df.loc[df['raw_drop_number'].astype(str).str.len() == 4096]

# # - Convert time column to datetime 
# df['time'] = dd.to_datetime(df['time'], format='%d/%m/%Y %H:%M:%S')


#---------------------------------------------------------------------------.
#### 8.3 Run following code portion without modifying anthing 
# - This portion of code represent what is done by read_L0_raw_file_list in L0_proc.py

# ## Keep only clean data 
# # - This type of filtering will be done in the background automatically ;) 
# # Remove rows with bad data 
# # df = df[df.sensor_status == 0] 
# # Remove rows with error_code not 000 
# # df = df[df.error_code == 0]  

##----------------------------------------------------.
# Cast dataframe to dtypes
# - Determine dtype based on standards 
# dtype_dict = get_L0_dtype_standards()
# for column in df.columns:
#     try:
#         df[column] = df[column].astype(dtype_dict[column])
#     except KeyError:
#         # If column dtype is not into get_L0_dtype_standards, assign object
#         df[column] = df[column].astype('object')
        
#---------------------------------------------------------------------------.
#### 8.4 Check the dataframe looks as desired 
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

def df_sanitizer_fun(df, lazy=lazy):
    # Import dask or pandas 
    if lazy: 
        import dask.dataframe as dd
    else: 
        import pandas as dd
    
    # - Drop all nan in latitude (define in reader_kwargs['na_values'])
    df = df[~df.iloc[:,1].isna()]
    if len(df.index) == 0:
        df.columns = column_names
        return df
    
    # - If first column is ID, than is a different format
    
    if lazy:
        flag = df.iloc[:,0].str.isnumeric().all().compute()
    else:
        flag = df.iloc[:,0].str.isnumeric().all()
    
    if flag:
        # - Rename columns
        df.columns = column_names_2
        # - Remove ok from rainfall_rate_32bit
        if lazy:
            df["rainfall_rate_32bit"] = df["rainfall_rate_32bit"].str.replace("OK,","")
        else:
            # df['rainfall_rate_32bit'] = df['rainfall_rate_32bit'].str.split(',').str[-1]
           # - Suppress SettingWithCopyWarning error (A value is trying to be set on a copy of a slice from a DataFrame)
           pd.options.mode.chained_assignment = None
           df['rainfall_rate_32bit'] = df['rainfall_rate_32bit'].str.split(',').str[-1]
        
        # - Drop useless columns
        col_to_drop = ["id", "all_nan", "All_0", 'datalogger_error', 'End_line']
        df = df.drop(columns=col_to_drop)
        
        # - Check latutide and longitute
        df = df.loc[df["latitude"].astype(str).str.len() < 11]
        df = df.loc[df["longitude"].astype(str).str.len() < 11]
        
        # - Convert time column to datetime 
        df['time'] = dd.to_datetime(df['time'], errors='coerce')
        df = df.dropna()
        if len(df.index) == 0:
            return df
        df['time'] = dd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
    else:
        # - Drop excedeed columns
        df = df.iloc[:,:18]
        # - Rename columns
        df.columns = column_names
        # - Drop useless columns
        col_to_drop = ["All_0", 'datalogger_error']
        df = df.drop(columns=col_to_drop)
        # - Convert time column to datetime 
        df['time'] = dd.to_datetime(df['time'], errors='coerce')
        df = df.dropna()
        if len(df.index) == 0:
            return df
        df['time'] = dd.to_datetime(df['time'], format='%d/%m/%Y %H:%M:%S')

    # - Drop columns if nan
    col_to_drop_if_na = ['latitude','longitude','raw_drop_concentration','raw_drop_average_velocity','raw_drop_number']
    df = df.dropna(subset = col_to_drop_if_na)
    
    # - Cast dataframe to dtypes
    dtype_dict = get_L0_dtype_standards(sensor_name=sensor_name)
    
    dtype_dict_not_object = {}
    for k, v in dtype_dict.items():
        if v != 'object':
            dtype_dict_not_object[k] =  v
    dtype_dict_not_object.pop('time')
            
    for column in df.columns:
        if column in dtype_dict_not_object:
            df[column] = dd.to_numeric(df[column], errors='coerce')
            invalid_rows_index = df.loc[df[column].isna()].index
            if lazy:
                if invalid_rows_index.size.compute() != 0:
                    df = df.dropna(subset=[column])
            else:
                if invalid_rows_index.size != 0:
                    df = df.dropna(subset=[column])
                    # df = df.drop(invalid_rows_index)
            df[column] = df[column].astype(dtype_dict[column])

    
    return df 

##------------------------------------------------------. 
#### 9.2 Launch code as in the parser file 
# - Try with increasing number of files 
# - Try first with lazy=False, then lazy=True 
lazy = False # True 
subset_file_list = file_list[20:27]
# subset_file_list = all_stations_files
df = read_L0_raw_file_list(file_list=subset_file_list, 
                           column_names=temp_column_names, 
                           reader_kwargs=reader_kwargs,
                           sensor_name=sensor_name,
                           verbose=verbose,
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
parquet_dir = os.path.join(processed_dir, 'L0', campaign_name + '.parquet')

# Define writing options 
compression = 'snappy' # 'gzip', 'brotli, 'lz4', 'zstd'
row_group_size = 100000 
engine = "pyarrow"

df = df.to_parquet(parquet_dir , 
                    # schema = 'infer',
                    engine = engine,
                    row_group_size = row_group_size,
                    compression = compression
                  )
##------------------------------------------------------. 
#### 10.1 Read parquet file
df = dd.read_parquet(parquet_dir)
df = df.compute()
print(df)

####--------------------------------------------------------------------------. 
##------------------------------------------------------. 
#### 11. L1 processing
#---------------------------------------------------------------------. 
# Retrieve metadata 
attrs = read_metadata(raw_dir=raw_dir,
                      station_id=station_id)
# Retrieve sensor name
sensor_name = attrs['sensor_name']
#-----------------------------------------------------------------.
#### - Create xarray Dataset
ds = create_L1_dataset_from_L0(df=df, attrs=attrs, lazy=lazy, verbose=verbose)
