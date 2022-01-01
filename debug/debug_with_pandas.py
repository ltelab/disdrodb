#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 21:53:31 2021

@author: ghiggi
"""
import glob 
import shutil
import pandas as pd 
import dask.dataframe as dd
import dask.array as da
import numpy as np 
import xarray as xr

import disdrodb.io 
lazy = True 

# Define reader_kwargs 
reader_kwargs = {}
 
# reader_kwargs['compression'] = 'gzip'
reader_kwargs['delimiter'] = ','
reader_kwargs["on_bad_lines"] = 'skip'
reader_kwargs["engine"] = 'python'
reader_kwargs["index_col"] = False
reader_kwargs["blocksize"] = None

# - Replace custom NA with standard flags 
reader_kwargs['na_values'] = 'na'
 
# Read data
df = read_raw_data(filepath=filename, 
                   column_names=raw_data_columns,
                   reader_kwargs=reader_kwargs,
                   lazy=lazy)
if lazy:
    df = df.compute()
    
# Print columns 
df.columns

# Print each first 3 row of each column 
for column in df.columns:
    print("Column:", column)
    print(df[column].iloc[0:5])

#------------------------------------------------------. 
# Compare to expected values 
# TODO:
from disdrodb.io import col_dtype_check
col_dtype_check(df, filename, verbose)
                  
##----------------------------------------------------.
def print_unique(df):
    '''
    Return all unique the unique values of a dataframe into a dictionary
    '''
    a = {}

    for col in list(df):
        a[col] = df[col].unique()
        
    for key, value in a.items():
        print(key, ' : ', value)
        
def print_nan_rows(df):
    is_NaN = df.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = df[row_has_NaN]

    print(rows_with_NaN)


#------------------------------------------------------. 
# Check if file empty
if len(df.index) == 0:
    raise ValueError(f"{filename} is empty and has been skipped.")

# Check column number
if len(df.columns) != len(raw_data_columns):
    raise ValueError(f"{filename} has wrong columns number, and has been skipped.")

#------------------------------------------------------.  
##########################
#### Custom code here ####
##########################
# --> To be enwrapped into parser custom df_sanitizer_fun !!!!            
            
# Split erroneous columns # TODO 
# df_tmp = df['TO_BE_SPLITTED'].astype(str).str.split(',', n=1, expand=True)
# df_tmp.columns = ['datalogger_voltage','rain_rate_32bit']
# df = df.drop(columns=['TO_BE_SPLITTED'])
# df = dd.concat([df, df_tmp], axis = 1, ignore_unknown_divisions=True)
# del df_tmp 

# Drop latitude and longitute (always the same)
df = df.drop(columns=['latitude', 'longitude'])

# Drop rows with half nan values
df = df.dropna(thresh = (len(df.columns) - 10), how = 'all')

#------------------------------------------------------.
##########################
#### Fixed code here #####
##########################
### Keep only clean data 
# - This will be done in the background ;) 
# Remove rows with bad data 
df = df[df.sensor_status == 0] 

# Remove rows with error_code not 000 
df = df[df.error_code == 0]  

##----------------------------------------------------.
# Cast dataframe to dtypes
# - Determine dtype based on standards 
dtype_dict = get_L0_dtype_standards()
dtype_dict = {column: dtype_dict[column] for column in df.columns}
for k, v in dtype_dict.items():
    df[k] = df[k].astype(v)
    

