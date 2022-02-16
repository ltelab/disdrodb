#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:26:01 2022

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import os
import xarray as xr
import netCDF4

# from disdrodb.data_encodings import get_ARM_to_l0_dtype_standards
from disdrodb.standards import get_var_explanations_ARM

def compare_standard_keys(dict_campaing, ds_keys):
    
    dict_standard = {}
    
    
    for ds_v in ds_keys:
        for dict_k, dict_v in dict_campaing.items():
            if dict_k == ds_v:
                dict_standard[dict_k] = dict_v
                continue
        dict_standard[ds_v] = ds_v + '_OldName'
        
    # for dict_k, dict_v in dict_campaing.items():
    #     for ds_v in ds_keys:
    #         if dict_k == ds_v:
    #             dict_standard[dict_k] = dict_v
    #             continue
    #     dict_standard[ds_v] = ds_v + '_OldName'
    
    return dict_standard

dict_ARM =      {'time': 'time',
                'base_time': 'time',
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

campagna = 'nsalpmC1.a1.20170429.000800'
path = "/SharedVM/Campagne/ARM/Raw/ALASKA/data/nsalpmC1"
file = campagna + '.nc'
file_path = os.path.join(path, file)

ds = xr.open_dataset(file_path)

print(list(ds.keys()))

print(list(ds.data_vars))


ds2 = xr.open_dataset('/SharedVM/Campagne/ARM/Processed/ALASKA/L1/ALASKA_nsalpmC1_0')
print(list(ds2.keys()))
ds2.close()

import netCDF4
import numpy as np
f = netCDF4.Dataset(file_path)
print(f.__dict__)

asd = f.variables

print(f.__dict__)

# ds_keys = []

# for k in ds.keys():
#     ds_keys.append(k)
    
# asd = compare_standard_keys(dict_ARM, ds_keys)

# ds = ds.rename(dict_ARM)

# print(list(ds.keys()))


# ds.to_netcdf("/SharedVM/Campagne/ARM/Processed/NORWAY/L0/lol.nc", mode='w', format="NETCDF4")


ds.close()






