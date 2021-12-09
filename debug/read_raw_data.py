#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:09:19 2021

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd

print(' ### INIZIO ### ')
print()

path = r'/SharedVM/Campagne/ltnas3'
campagna = 'Ticino_2018'
folder_device = '61'
file_name = 'Station-61-2019-07-14.dat'

path_raw = path + r'/Raw/' + campagna + '/data/' + folder_device + '/' + file_name
path_processed = path + r'/processed/' + campagna + '/30/' + campagna + '.parquet'

raw_data_columns = ['id',
                    'latitude',
                    'longitude',
                    'time',
                    'datalogger_temperature',
                    'datalogger_voltage',
                    'rain_rate_32bit',
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
                    'error_code',
                    'FieldN',
                    'FieldV',
                    'RawData',
                    'datalogger_error',
                    ]

n = 0
col_names = {}
for name in raw_data_columns:
    col_names[n] = name
    n += 1
    


df = pd.read_csv(path_raw, delimiter = ',,', names = ['a','FieldV','RawData','datalogger_error'], engine = 'python', na_values = 'na')

# df2 = df['a'].str.split(',', expand = True, n = 20)

df = pd.concat([df['a'].str.split(',', expand = True, n = 20),df.iloc[:,1:3]], axis = 1).rename(columns = col_names)



