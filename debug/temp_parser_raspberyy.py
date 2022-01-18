#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:16:28 2022

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
raw_dir = "/SharedVM/Campagne/DELFT/Raw/TEST_DATA"
processed_dir = "/SharedVM/Campagne/DELFT/Processed/TEST_DATA"

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
glob_pattern = os.path.join("data", station_id, "*.csv*") # CUSTOMIZE THIS 
device_path = os.path.join(raw_dir, glob_pattern)
file_list = sorted(glob.glob(device_path, recursive = True))
#-------------------------------------------------------------------------. 
# All files into the campaing
all_stations_files = sorted(glob.glob(os.path.join(raw_dir, "data", "*/*.csv*"), recursive = True))
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
reader_kwargs['na_values'] = ['na', '', 'error', 'NA', '-.-', ' NA',]

# - Define max size of dask dataframe chunks (if lazy=True)
#   - If None: use a single block for each file
#   - Otherwise: "<max_file_size>MB" by which to cut up larger files
reader_kwargs["blocksize"] = None # "50MB" 

# Cast all to string
reader_kwargs["dtype"] = str

# Skip first row as columns names
reader_kwargs['header'] = None

# Use for Nan value
# reader_kwargs['assume_missing'] = True

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
                    lazy=True).add_prefix('col_')

# Add prefix to columns
# df = df.add_prefix('col_')
# df_to_parse = df_to_parse.compute()
# df = df.compute()

# Split the last column (contain the 37 remain fields)
df_to_parse = df['col_2'].str.split(';', expand=True, n = 99).add_prefix('col_')


df['col_0'] = dd.to_datetime(df['col_0'], format='%Y%m%d-%H%M%S')

# Split latidude and longitude
df[['latidude', 'longitude']] = df['col_1'].str.split(pat=".", expand=True, n = 1)
df3 = df['col_1'].str.split(pat=".", expand=True, n = 2).compute()

# Drop unused columns
# df = df.drop(['col_1', 'col_2'], axis=1)

# Remove char from rain intensity
df_to_parse['col_0'] = df_to_parse['col_0'].str.lstrip("b'")


# Add the comma on the FieldN, FieldV and RawData
df_FieldN = df_to_parse.iloc[:,36:67].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1, meta=(None, 'object'))
df_FieldV = df_to_parse.iloc[:,68:-1].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1, meta=(None, 'object'))
df_RawData = df_to_parse.iloc[:,-1:].squeeze().str.replace(r'(\w{3})', r'\1,', regex=True).str.rstrip("'")

# Concat all togheter
df = dd.concat([df, df_to_parse.iloc[:,:35], df_FieldN, df_FieldV, df_RawData] ,axis=1)

# Print first rows
print_df_first_n_rows(df, n = 1, column_names=False)
print_df_first_n_rows(df, n = 5, column_names=False)
print_df_random_n_rows(df, n= 5, column_names=False)  # this likely the more useful 

# Retrieve number of columns 
print(len(df.columns))
 
# Look at unique values
# print_df_columns_unique_values(df, column_indices=None, column_names=False) # all 
 
# print_df_columns_unique_values(df, column_indices=0, column_names=False) # single column 

# print_df_columns_unique_values(df, column_indices=slice(0,15), column_names=False) # a slice of columns 

# get_df_columns_unique_values_dict(df, column_indices=slice(0,15), column_names=False) # get dictionary

# Retrieve number of columns 
print(len(df.columns))
 
# Copy the following list and start to infer column_names 
['Unknown' + str(i+1) for i in range(len(df.columns))]

# Print valid column names 
# - If other names are required, add the key to get_L0_dtype_standards in data_encodings.py 
print_valid_L0_column_names()

# Instrument manufacturer defaults 
from disdrodb.standards import get_OTT_Parsivel_dict, get_OTT_Parsivel2_dict
get_OTT_Parsivel_dict()
get_OTT_Parsivel2_dict()


####---------------------------------------------------------------------------.
######################################################################
#### 7. Define dataframe columns [TO CUSTOMIZE AND MOVE TO PARSER] ###
######################################################################
# - If a column must be splitted in two (i.e. lat_lon), use a name like: TO_SPLIT_lat_lon

# "01","Rain intensity 32 bit",8,"mm/h","single_number"
# "02","Rain amount accumulated 32 bit",7,"mm","single_number"
# "03","Weather code SYNOP Table 4680",2,"","single_number"
# "04","Weather code SYNOP Table 4677",2,"","single_number"
# "05","Weather code METAR Table 4678",5,"","character_string"
# "06","Weather code NWS",4,"","character_string"
# "07","Radar reflectivity 32 bit",6,"dBZ","single_number"
# "08","MOR visibility in precipitation",5,"m","single_number"
# "09","Sample interval",5,"s","single_number"
# "10","Signal amplitude of laser",5,"","single_number"
# "11","Number of particles detected and validated",5,"","single_number"
# "12","Temperature in sensor housing",3,"degree_Celsius","single_number"
# "13","Sensor serial number",6,"","character_string"
# "14","Firmware IOP",6,"","character_string"
# "15","Firmware DSP",6,"","character_string"
# "16","Sensor head heating current",4,"A","single_number"
# "17","Power supply voltage",4,"V","single_number"
# "18","Sensor status",1,"","single_number"
# "19","Date/time measuring start",19,"DD.MM.YYYY_hh:mm:ss","character_string"
# "20","Sensor time",8,"hh:mm:ss","character_string"
# "21","Sensor date",10,"DD.MM.YYYY","character_string"
# "22","Station name",4,"","character_string"
# "23","Station number",4,"","character_string"
# "24","Rain amount absolute 32 bit",7,"mm","single_number"
# "25","Error code",3,"","character_string"
# "26","Temperature PCB",3,"degree_Celsius","single_number"
# "27","Temperature in right sensor head",3,"degree_Celsius","single_number"
# "28","Temperature in left sensor head",3,"degree_Celsius","single_number"
# "30","Rain intensity 16 bit max 30 mm/h",6,"mm/h","single_number"
# "31","Rain intensity 16 bit max 1200 mm/h",6,"mm/h","single_number"
# "32","Rain amount accumulated 16 bit",7,"mm","single_number"
# "33","Radar reflectivity 16 bit",5,"dBZ","single_number"
# "34","Kinetic energy",7,"J/(m2*h)","single_number"
# "35","Snowfall intensity",7,"mm/h","single_number"
# "60","Number of all particles detected",8,"","single_number"
# "61","List of all particles detected",13,"","list"
# "90","FieldN",224,"","vector"
# "91","FieldV",224,"","vector"
# "93","Raw data",4096,"","matrix"

columns_names_temporary =['time','TO_BE_SPLITTED','TO_BE_PARSED']

column_names = ['time',
                'unknow',
                'unknow2',
                'rain_rate_32bit',
                'rain_accumulated_32bit',
                'weather_code_SYNOP_4680',
                'weather_code_SYNOP_4677',
                'weather_code_METAR_4678',
                'weather_code_NWS',
                'reflectivity_32bit',
                'mor_visibility',
                'sample_interval',
                'laser_amplitude',
                'n_particles',
                'sensor_temperature',
                'sensor_serial_number',
                'firmware_IOP',
                'firmware_DSP',
                'sensor_heating_current',
                'sensor_battery_voltage',
                'sensor_status',
                'date_time_measuring_start',
                'sensor_time',
                'sensor_date',
                'station_name',
                'station_number',
                'rain_amount_absolute_32bit',
                'datalogger_error',
                'datalogger_temperature',
                'sensor_temperature_right',
                'sensor_temperature_left',
                'rain_rate_16bit',
                'rain_rate_12bit',
                'rain_accumulated_16bit',
                'reflectivity_16bit',
                'rain_kinetic_energy',
                'snowfall_intensity',
                'n_particles_all',
                'n_particles_all_detected',
                'FieldN',
                'FieldV',
                'RawData',
                ]

# - Check name validity 
check_L0_column_names(column_names)

# - Read data
# Added function read_raw_data_dtype() on L0_proc for read with columns and all dtypes as object
filepath = file_list[0]
df = read_raw_data(filepath=filepath, 
                    column_names=columns_names_temporary,
                    reader_kwargs=reader_kwargs,
                    lazy=False)
    


# - Look at the columns and data 
print_df_column_names(df)
print_df_random_n_rows(df, n= 5)

# - Check it loads also lazily in dask correctly
df1 = read_raw_data(filepath=filepath, 
                    column_names=column_names,
                    reader_kwargs=reader_kwargs,
                    lazy=True)

df1 = df1.compute() 

# - Look at the columns and data 
print_df_column_names(df1)
print_df_random_n_rows(df1, n= 5) 

# - Check are equals 
assert df.equals(df1)

# - Look at unique values
print_df_columns_unique_values(df, column_indices=None, column_names=True) # all 
 
print_df_columns_unique_values(df, column_indices=0, column_names=True) # single column 

print_df_columns_unique_values(df, column_indices=slice(0,10), column_names=True) # a slice of columns 

get_df_columns_unique_values_dict(df, column_indices=slice(0,15), column_names=True) # get dictionary

####---------------------------------------------------------------------------.
#########################################################
#### 8. Implement ad-hoc processing of the dataframe ####
#########################################################
# - This must be done once that reader_kwargs and column_names are correctly defined 
# - Try the following code with various file and with both lazy=True and lazy=False 
filepath = file_list[0]  # Select also other files here  1,2, ... 
lazy = False             # Try also with True when work with False 

#------------------------------------------------------. 
#### 8.1 Run following code portion without modifying anthing 
# - This portion of code represent what is done by read_L0_raw_file_list in L0_proc.py
df = read_raw_data(filepath=filepath, 
                    column_names=columns_names_temporary,
                    reader_kwargs=reader_kwargs,
                    lazy=lazy)

#------------------------------------------------------. 
# Check if file empty
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
# df_tmp.columns = ['datalogger_voltage','rain_rate_32bit']
# df = df.drop(columns=['TO_BE_SPLITTED'])
# df = dd.concat([df, df_tmp], axis = 1, ignore_unknown_divisions=True)
# del df_tmp 

# Add prefix to columns
# df = df.add_prefix('col_')

# ----

# Split the last column (contain the 37 remain fields)
df_to_parse = df['TO_BE_PARSED'].str.split(';', expand=True, n = 99)

# Cast to datetime
df['time'] = dd.to_datetime(df['time'], format='%Y%m%d-%H%M%S')

# Split latidude and longitude and drop TO_BE_SPLITTED and TO_BE_PARSED
df[['latidude', 'longitude']] = df['TO_BE_SPLITTED'].str.split(pat=".", expand=True, n = 1)
df = df.drop(['TO_BE_SPLITTED', 'TO_BE_PARSED'], axis=1)

# Add names to columns
df_to_parse_dict_names = dict(zip(column_names[3:-3],list(df_to_parse.columns)[0:35]))
for i in range(len(list(df_to_parse.columns)[35:])):
    df_to_parse_dict_names[i] = i

df_to_parse.columns = df_to_parse_dict_names

# Remove char from rain intensity
df_to_parse['rain_rate_32bit'] = df_to_parse['rain_rate_32bit'].str.lstrip("b'")

# Remove spaces on weather_code_METAR_4678 and weather_code_NWS
df_to_parse['weather_code_METAR_4678'] = df_to_parse['weather_code_METAR_4678'].str.strip()
df_to_parse['weather_code_NWS'] = df_to_parse['weather_code_NWS'].str.strip()

# ----

# Add the comma on the FieldN, FieldV and RawData
df_FieldN = df_to_parse.iloc[:,35:67].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1).to_frame('FieldN')
df_FieldV = df_to_parse.iloc[:,67:-1].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1).to_frame('FieldV')
df_RawData = df_to_parse.iloc[:,-1:].squeeze().str.replace(r'(\w{3})', r'\1,', regex=True).str.rstrip("'").to_frame('RawData')

# Concat all togheter
df = dd.concat([df, df_to_parse.iloc[:,:35], df_FieldN, df_FieldV, df_RawData] ,axis=1)

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
dtype_dict = get_L0_dtype_standards()
for column in df.columns:
    try:
        df[column] = df[column].astype(dtype_dict[column])
    except KeyError:
        # If column dtype is not into get_L0_dtype_standards, assign object
        df[column] = df[column].astype('object')
    except ValueError as e:
        print(f"The column {column} has {e}")
        
#---------------------------------------------------------------------------.
#### 8.4 Check the dataframe looks as desired 
print_df_column_names(df)
print_df_random_n_rows(df, n= 5) 
print_df_columns_unique_values(df, column_indices=2, column_names=True) 
print_df_columns_unique_values(df, column_indices=slice(0,20), column_names=True)   

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
    
    # Split the last column (contain the 37 remain fields)
    df_to_parse = df['TO_BE_PARSED'].str.split(';', expand=True, n = 99)

    # Cast to datetime
    df['time'] = dd.to_datetime(df['time'], format='%Y%m%d-%H%M%S')

    # Split latidude and longitude and drop TO_BE_SPLITTED and TO_BE_PARSED
    df[['unknow', 'unknow2']] = df['TO_BE_SPLITTED'].str.split(pat=".", expand=True, n = 1)
    df = df.drop(['TO_BE_SPLITTED', 'TO_BE_PARSED'], axis=1)

    # Add names to columns
    df_to_parse_dict_names = dict(zip(column_names[3:-3],list(df_to_parse.columns)[0:35]))
    for i in range(len(list(df_to_parse.columns)[35:])):
        df_to_parse_dict_names[i] = i

    df_to_parse.columns = df_to_parse_dict_names

    # Remove char from rain intensity
    df_to_parse['rain_rate_32bit'] = df_to_parse['rain_rate_32bit'].str.lstrip("b'")

    # Remove spaces on weather_code_METAR_4678 and weather_code_NWS
    df_to_parse['weather_code_METAR_4678'] = df_to_parse['weather_code_METAR_4678'].str.strip()
    df_to_parse['weather_code_NWS'] = df_to_parse['weather_code_NWS'].str.strip()

    # ----

    # Add the comma on the FieldN, FieldV and RawData
    df_FieldN = df_to_parse.iloc[:,35:67].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1, meta=(None, 'object')).to_frame('FieldN')
    df_FieldV = df_to_parse.iloc[:,67:-1].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1, meta=(None, 'object')).to_frame('FieldV')
    df_RawData = df_to_parse.iloc[:,-1:].squeeze().str.replace(r'(\w{3})', r'\1,', regex=True).str.rstrip("'").to_frame('RawData')

    # Concat all togheter
    df = dd.concat([df, df_to_parse.iloc[:,:35], df_FieldN, df_FieldV, df_RawData] ,axis=1, ignore_unknown_divisions=True)
    
    return df 

##------------------------------------------------------. 
#### 9.2 Launch code as in the parser file 
# - Try with increasing number of files 
# - Try first with lazy=False, then lazy=True 
lazy = True # True 
subset_file_list = file_list[:1]
# subset_file_list = all_stations_files
df = read_L0_raw_file_list(file_list=subset_file_list, 
                           column_names=columns_names_temporary, 
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









































