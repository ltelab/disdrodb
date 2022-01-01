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

def get_L0_dtype_standards(): 
    dtype_dict = {                                 
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
        "weather_code_NWS":'object',    #TODO
        
        "n_particles"     :'uint32',
        "n_particles_all": 'uint32',
        
        "sensor_temperature": 'uint8',
        "temperature_PBC" : 'object',   #TODO
        "temperature_right" : 'object', #TODO
        "temperature_left":'object',    #TODO
        
        "sensor_heating_current" : 'float32',
        "sensor_battery_voltage" : 'float32',
        "datalogger_heating_current" : 'float32',
        "datalogger_battery_voltage" : 'float32',
        "sensor_status"   : 'uint8',
        "laser_amplitude" :'uint32',
        "error_code"      : 'uint8',          

        # Custom fields       
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
        'TEMPORARY': 'object',
        'TO_BE_PARSED': 'object',
        'TO_BE_SPLITTED': 'object',
        'TO_DEBUG': 'object',
        "Debug_data" : 'object',
        'All_0': 'object',
        'error_code?': 'object',
        'unknow2': 'object',
        'unknow3': 'object',
        'unknow4': 'object',
        'unknow5': 'object',
        'unknow': 'object',
        'unknow6': 'object',
        'unknow7': 'object',
        'unknow8': 'object',
        'unknow9': 'object',
        'power_supply_voltage': 'object',
        'A_voltage2?' : 'object',
        'A_voltage?' : 'object',
        'All_nan' : 'object',
        'All_5000' : 'object',
        
    }
    return dtype_dict

def get_L1_dtype():
    # Float 32 or Float 64 (f4, f8)
    # (u)int 8 16, 32, 64   (u/i  1 2 4 8)
    dtype_dict = {'FieldN': 'float32',
                  'FieldV': 'float32',  
                  'RawData': 'int64',   # TODO: uint16? uint32 check largest number occuring, and if negative
                 }
    return dtype_dict

def get_dtype_standards_all_object(): 
    dtype_dict = get_L0_dtype_standards()
    for i in dtype_dict:
        dtype_dict[i] = 'object'
        
    return dtype_dict

