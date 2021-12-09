#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:09:19 2021

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np

print(' ### INIZIO ### ')
print()

path = r'/SharedVM/Campagne/ltnas3'
campagna = 'Ticino_2018'
folder_device = '61'
file_name = 'Station-61-2020-07-23.dat'

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
    


df = pd.read_csv(path_raw, delimiter = ',,', names = ['a','FieldV','RawData','datalogger_error'], engine = 'python', na_values = ['na', 'Error in data reading! error0', 'None'])

# df2 = df['a'].str.split(',', expand = True, n = 20)

df = pd.concat([df['a'].str.split(',', expand = True, n = 20),df.iloc[:,1:3]], axis = 1).rename(columns = col_names)

# df[df['rain_rate_32bit'].apply(lambda x: str(x).isdigit())]

df = df.drop(columns=['latitude', 'longitude'])

df = df.dropna(thresh = 15)

# df = df.fillna(value=np.nan)

# ---------------------------------------------

dtype_dict = {                                 # Kimbo option
    "id": "uint32",
    "rain_rate_16bit": 'float32',
    "rain_rate_32bit": 'object',
    "rain_accumulated_16bit":   'float32',
    "rain_accumulated_32bit":   'float32',
    
    "rain_amount_absolute_32bit": 'float32', 
    "reflectivity_16bit": 'float32',
    "reflectivity_32bit": 'float32',
    
    "rain_kinetic_energy"  :'float32',
    "snowfall_intensity": 'float32',
    
    "mor_visibility"    :'uint16',
    
    "weather_code_SYNOP_4680":'uint8',             
    "weather_code_SYNOP_4677":'uint8',              
    "weather_code_METAR_4678":'object', #TODO
    "weather_code_NWS":'object', #TODO
    
    "n_particles"     :'uint32',
    "n_particles_all": 'uint32',
    
    "sensor_temperature": 'uint8',
    "temperature_PBC" : 'object', #TODO
    "temperature_right" : 'object', #TODO
    "temperature_left":'object', #TODO
    
    "sensor_heating_current" : 'float32',
    "sensor_battery_voltage" : 'float32',
    "sensor_status"   : 'uint8',
    "laser_amplitude" :'uint32',
    "error_code"      : 'uint8',          

    # Custom ields       
    "Unknow_column": "object",
    "datalogger_temperature": "object",
    "datalogger_voltage": "object",
    'datalogger_error': 'uint8',
    
    # Data fields (TODO) 
    "FieldN": 'object',
    "FieldV": 'object',
    "RawData": 'object',
    
    # Coords 
    "latitude" : 'float32',
    "longitude" : 'float32',
    "altitude" : 'float32',
    
     # Dimensions
    'time': 'object',
    
    #Temp variables
    "Debug_data" : 'object',
    'All_0': 'object',
    'error_code?': 'object',
    'unknow2': 'object',
    'unknow3': 'object',
    'unknow4': 'object',
    'unknow5': 'object',
    'unknow': 'object',
    'unknow6': 'object',
    'power_supply_voltage': 'object',
    'A_voltage2?' : 'object'
    
}


# a = {}

# for col in list(df.iloc[:,:18]):
#     a[col] = df[col].unique()
    
# for key, value in a.items():
#     print(key, ' : ', value)
    



dtype_dict = {column: dtype_dict[column] for column in df.columns}

for k, v in dtype_dict.items():
    print(f'{k} is {v}')
    df[k] = df[k].astype(v)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    