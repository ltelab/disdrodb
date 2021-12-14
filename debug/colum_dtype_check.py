#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:43:35 2021

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np

print()

# D:\SharedVM\Campagne\ltnas3\Raw\Ticino_2018\epfl_parsivel\raw\2019\07

# path = r'/SharedVM/Campagne/ltnas3/Raw/Ticino_2018/epfl_parsivel/raw/2019'
# campagna = 'Ticino_2018'
# folder_device = '07'
# file_name = 'Station-61-2019-07-13.dat'

# path_raw = path + r'/Raw/' + campagna + '/data/' + folder_device + '/' + file_name
# path_processed = path + r'/processed/' + campagna + '/30/' + campagna + '.parquet'

# path_raw = path + '/' + folder_device + '/' + file_name

path_raw = '/SharedVM/Campagne/ltnas3/Raw/debug/1.dat'

raw_data_columns = ['id',
                    'latitude',
                    'longitude',
                    'time',
                    'datalogger_temperature',
                    'x', #datalogger_voltage rain_rate_32bit
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
                    'datalogger_error'
                    ]


dtype_dict = {                                 # Kimbo option
    "id": "uint32",
    "rain_rate_16bit": 'float32',
    "rain_rate_32bit": 'float32',
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
    'temp': 'object',
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

import datetime

dtype_range_values = {
        'id': [0, 4294967295],
        'rain_rate_16bit': [0, 9999.999],
        'rain_rate_32bit': [0, 9999.999],
        'rain_accumulated_16bit': [0, 300.00],
        'rain_accumulated_32bit': [0, 300.00],
        'rain_amount_absolute_32bit': [0, 999.999],
        'reflectivity_16bit': [-9.999, 99.999],
        'reflectivity_32bit': [-9.999, 99.999],
        'rain_kinetic_energy': [0, 999.999],
        'snowfall_intensity': [0, 999.999],
        'mor_visibility': [0, 20000],
        'weather_code_SYNOP_4680': [0, 99],
        'weather_code_SYNOP_4677': [0, 99],
        'n_particles': [0, 0],  #For debug, [0, 99999]
        'n_particles_all': [0, 8192],
        'sensor_temperature': [-99, 100],
        'temperature_PBC': [-99, 100],
        'temperature_right': [-99, 100],
        'temperature_left': [-99, 100],
        'sensor_heating_current': [0, 4.00],
        'sensor_battery_voltage': [0, 30.0],
        'sensor_status': [0, 3],
        'laser_amplitude': [0, 99999],
        'error_code': [0,3],
        'datalogger_temperature': [-99, 100],
        'datalogger_voltage': [0, 30.0],
        'datalogger_error': [0,3],
        
        'latitude': [-90000, 90000],
        'longitude': [-180000, 180000],
        
        'time': [datetime.datetime(1900, 1, 1), datetime.datetime.now()]
       }

dtype_max_digit ={
        'id': [8],  #Maybe to change in the future
        'rain_rate_16bit': [8],
        'rain_rate_32bit': [8],
        'rain_accumulated_16bit': [7],
        'rain_accumulated_32bit': [7],
        'rain_amount_absolute_32bit': [7],
        'reflectivity_16bit': [6],
        'reflectivity_32bit': [6],
        'rain_kinetic_energy': [7],
        'snowfall_intensity': [7],
        'mor_visibility': [4],
        'weather_code_SYNOP_4680': [2],
        'weather_code_SYNOP_4677': [2],
        'n_particles': [5],
        'n_particles_all': [8],
        'sensor_temperature': [3],
        'temperature_PBC': [3],
        'temperature_right': [3],
        'temperature_left': [3],
        'sensor_heating_current': [4],
        'sensor_battery_voltage': [4],
        'sensor_status': [1],
        'laser_amplitude': [5],
        'error_code': [1],
        'datalogger_temperature': [3],
        'datalogger_voltage': [4],
        'datalogger_error': [1],
        
        'latitude': [9],
        'longitude': [15],
        
        'time': [19],
        
        'FieldN': [225],
        'FieldV': [225],
        'RawData': [4097],
    }

time_col = ['time']

df = pd.read_csv(path_raw, delimiter = ';', names = raw_data_columns, parse_dates = time_col)

for col in df.columns:
    try:
        
        if not df[col].astype(str).str.len().max() <= dtype_max_digit[col][0]:
            print(f'{col} has more than %s' % dtype_max_digit[col][0])
            print(f'Error, the values {col} have too much digits (%s) in index: %s' % (dtype_max_digit[col][0], df.index[df[col].astype(str).str.len() >= dtype_max_digit[col][0]].tolist()))
            
        
        if not df[col].between(dtype_range_values[col][0], dtype_range_values[col][1]).all():
            print(f'Error, the values {col} in index are not in dtype range: %s' % df.index[df[col].between(dtype_range_values[col][0], dtype_range_values[col][1]) == False].tolist())
            
                   
    except KeyError:
        print(f'No range values for {col}, check ignored')
        pass
    except TypeError:
        print(f'{col} is object, check ignored')
        pass
    
    
    






# df2 = df['x'].str.split(',', n=1, expand=True)

# df2.columns = ['datalogger_voltage','rain_rate_32bit']

# df.astype({'datalogger_voltage': 'int32', }).dtypes


