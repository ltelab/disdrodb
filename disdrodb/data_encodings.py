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
        "n_particles_all": 'object',
        
        "sensor_temperature": 'uint8',
        "sensor_temperature_PBC" : 'object',   #TODO
        "sensor_temperature_right" : 'object', #TODO
        "sensor_temperature_left":'object',    #TODO
        
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
        'time': 'M8[s]',
        
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
        
        #Disdronet raspberry variables
        "sample_interval": "uint16",
        "sensor_serial_number": "object",
        "firmware_IOP": "object",
        "firmware_DSP": "object",
        "date_time_measuring_start": "object",
        "sensor_time": "object",
        "sensor_date": "object",
        "station_name": "object",
        "station_number": "object",
        "rain_rate_12bit": "object",
        'n_particles_all_detected': 'object'
    }
    return dtype_dict

#### Dictionary to convert ARM netcdf to L1 standard
# - Use to rename the ARM keys to L1 standard
def get_ARM_to_l0_dtype_standards(): 

    dict_ARM_to_l0={'time': 'time',
                    'time_offset': 'time_offset_OldName',
                    'precip_rate': 'rain_rate_32bit',
                    'qc_precip_rate': 'qc_precip_rate_OldName',
                    'weather_code': 'weather_code_SYNOP_4680',
                    'qc_weather_code': 'qc_weather_code_OldName',
                    'equivalent_radar_reflectivity_ott': 'reflectivity_32bit',
                    'qc_equivalent_radar_reflectivity_ott': 'qc_equivalent_radar_reflectivity_ott_OldName',
                    'number_detected_particles': 'n_particles',
                    'qc_number_detected_particles': 'qc_number_detected_particles_OldName',
                    'mor_visibility': 'mor_visibility_OldName',
                    'qc_mor_visibility': 'qc_mor_visibility_OldName',
                    'snow_depth_intensity': 'snow_depth_intensity_OldName',
                    'qc_snow_depth_intensity': 'qc_snow_depth_intensity_OldName',
                    'laserband_amplitude': 'laser_amplitude',
                    'qc_laserband_amplitude': 'qc_laserband_amplitude_OldName',
                    'sensor_temperature': 'sensor_temperature',
                    'heating_current': 'sensor_heating_current',
                    'qc_heating_current': 'qc_heating_current_OldName',
                    'sensor_voltage': 'sensor_battery_voltage',
                    'qc_sensor_voltage': 'qc_sensor_voltage_OldName',
                    'class_size_width': 'class_size_width_OldName',
                    'fall_velocity_calculated': 'fall_velocity_calculated_OldName',
                    'raw_spectrum': 'raw_spectrum_OldName',
                    'liquid_water_content': 'liquid_water_content_OldName',
                    'equivalent_radar_reflectivity': 'equivalent_radar_reflectivity_OldName',
                    'intercept_parameter': 'intercept_parameter_OldName',
                    'slope_parameter': 'slope_parameter_OldName',
                    'median_volume_diameter': 'median_volume_diameter_OldName',
                    'liquid_water_distribution_mean': 'liquid_water_distribution_mean_OldName',
                    'number_density_drops': 'number_density_drops_OldName',
                    'diameter_min': 'diameter_min_OldName',
                    'diameter_max': 'diameter_max_OldName',
                    'moment1': 'moment1_OldName',
                    'moment2': 'moment2_OldName',
                    'moment3': 'moment3_OldName',
                    'moment4': 'moment4_OldName',
                    'moment5': 'moment5_OldName',
                    'moment6': 'moment6_OldName',
                    'lat': 'latitude',
                    'lon': 'longitude',
                    'alt': 'altitude',
                    }
    
    return dict_ARM_to_l0

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

