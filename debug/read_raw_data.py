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

print(' ### INIZIO ### ')
print()

raw_dir = "/SharedVM/Campagne/ltnas3/Raw/EPFL_Roof_2008/data"

path = raw_dir
campagna = 'EPFL_Roof_2008'
folder_device = '03'
file_name = '03_ascii_20080314.dat'

path_raw = os.path.join(path, folder_device, file_name)
# path_processed = path + r'/processed/' + campagna + '/30/' + campagna + '.parquet'

# path_raw = path + '/' + folder_device + '/' + file_name

raw_data_columns = ['time',
                    'id',
                    'datalogger_temperature',
                    'datalogger_voltage',
                    'Unknow_column', #Intensity
                    'rain_accumulated_32bit',
                    'weather_code_SYNOP_4680',
                    'weather_code_SYNOP_4677',
                    'reflectivity_16bit',
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
                  delimiter = '","', 
                  names = raw_data_columns,
                  skiprows= 4,
                  nrows=10,
                  engine='python'
                  )


df['time'] = df['time'].str[1:]
df['RawData'] = df['RawData'].str[:-1]

# Drop Debug_data
df = df.drop(['Debug_data'], axis=1)

df = df.drop(['All_0'], axis=1)

# Drop rows with more than 8 nan
df = df.dropna()

df['id'] = df['id'].astype('uint32')
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    