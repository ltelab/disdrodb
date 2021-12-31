#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:09:19 2021

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
import os
import glob

print(' ### INIZIO ### ')
print()

raw_dir = "/SharedVM/Campagne/ltnas3/Raw/EPFL_Roof_2011/data"

path = raw_dir
campagna = 'EPFL_Roof_2011'
folder_device = '10'
file_name = '10_ascii_20110811.dat'
device_path = os.path.join(raw_dir, folder_device)

path_raw = os.path.join(path, folder_device, file_name)
# path_processed = path + r'/processed/' + campagna + '/30/' + campagna + '.parquet'

# path_raw = path + '/' + folder_device + '/' + file_name

raw_columns_names = ['time', 
						'id', 
						'datalogger_temperature', 
						'datalogger_voltage', 
						'unknow', 
						'rain_accumulated_32bit', 
						'weather_code_SYNOP_4680',
						'weather_code_SYNOP_4677',
						'reflectivity_16bit',
						'mor_visibility',
                        'laser_amplitude',
                        'n_particles',
                        'sensor_temperature',
                        'All_nan',
                        'sensor_heating_current',
                        'All_0'
                        'unknow2',
                        'rain_amount_absolute_32bit',
                        'Debug_data',
                        'FieldN',
                        'FieldV',
                        'RawData',
                        'End_line'
						]



##------------------------------------------------------.
# Define reader options 
# reader_kwargs = {}
# # reader_kwargs['compression'] = 'gzip'
# reader_kwargs['delimiter'] = ';'
# reader_kwargs["on_bad_lines"] = 'skip'
# reader_kwargs["engine"] = 'python'
# # - Replace custom NA with standard flags 
# reader_kwargs['na_values'] = 'na'
# reader_kwargs["blocksize"] = None


df = pd.read_csv(path_raw,
                  # delimiter = '","', 
                  # names = raw_data_columns,
                  # skiprows= 4,
                  # nrows=10,
                  na_values = ['-.-'],
                  engine='python',
                  header=None,
                  ).add_prefix('col_')


df = df.drop(columns = ['col_13','col_17','col_21'])


# df['time'] = df['time'].str[1:]
# df['RawData'] = df['RawData'].str[:-1]

# Drop Debug_data
# df = df.drop(['Debug_data'], axis=1)

# df = df.drop(['All_0'], axis=1)

# Drop rows with more than 8 nan
# df = df.dropna()

# df['id'] = df['id'].astype('uint32')
# # Drop rows with more than 2 nan (longitute and latitude)
# df = df.dropna(thresh = (len(raw_data_columns) - 2), how = 'all')

# # Drop all 0 column
# df = df.drop(columns = ['All_0'])

# # Split Debug_data
# df1 = df['Debug_data'].str.split(r'T ', expand=True, n = 13).add_prefix('col_')

# df1 = df1.drop(['col_0'], axis=1)

# df2 = df1['col_3'].str.rsplit(r'  ', expand=True, n = 6).add_prefix('col_')

# df2 = df2['col_1'] + df2['col_2'] + df2['col_3'] + df2['col_4'] + df2['col_5'] + df2['col_6']

# df1 = df1.drop([0,1,7], axis = 1)
# df1.columns = ["Debug_data1","Debug_data2","Debug_data3","Debug_data4","Debug_data5"]
# df = dd.concat([df, df1], axis = 1, ignore_unknown_divisions=True)
# df = df.drop(['Debug_data'], axis = 1)


    
# Define reader options 
reader_kwargs = {}
reader_kwargs["engine"] = 'python'
# - Replace custom NA with standard flags 
reader_kwargs['na_values'] = ['', 'error', 'NA', 'na', '-.-']
# Define time column
# reader_kwargs['parse_dates'] = time_col
reader_kwargs["blocksize"] = None
reader_kwargs['header'] = None
reader_kwargs['encoding'] = 'latin-1'  # Important for this campaign
reader_kwargs['assume_missing'] = True
    
# ---------- Multiple dat reader ------------

file_list = sorted(glob.glob(os.path.join(device_path, "**/*.dat*"), recursive = True))
    
# file_list = file_list[0:3] 

list_df = []

for file in file_list:
    
    try:
        
        df = dd.read_csv(filename,
                        names = raw_data_columns,
                        dtype=dtype_dict,
                        **reader_kwargs
                        )
    
        # df = dd.read_csv(file,
        #                   names = raw_columns_names,
        #                   na_values = ['-.-', 'NA'],
        #                   engine='python',
        #                   header=None,
        #                    encoding='latin-1', # Important for this campaign
        #                   assume_missing=True,
        #                   dtype=dtype #all object and the to cast
        #                   )
        
        df = df.drop(columns = ['All_nan','Debug_data','End_line'])
        
        # df = df.dropna()
        
        # - Append to the list of dataframe 
        list_df.append(df)
    except Exception as e:
        print(file)
        print(e)
        pass
    
df = dd.concat(list_df, axis=0, ignore_index = True)
    
df = df.compute()
    
    
    
    
    
    