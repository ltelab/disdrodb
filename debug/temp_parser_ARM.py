#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 02:45:20 2022

@author: kimbo
"""

import os
import logging
import glob 
import shutil
import pandas as pd 
import dask.dataframe as dd
import dask.array as da
import numpy as np 
import xarray as xr
import netCDF4

from disdrodb.io import check_directories
from disdrodb.io import get_campaign_name
from disdrodb.io import create_directory_structure
from disdrodb.L0_proc import read_raw_data
from disdrodb.L0_proc import get_file_list
from disdrodb.logger import create_logger
from disdrodb.data_encodings import get_ARM_LPM_dict
from disdrodb.data_encodings import get_ARM_LPM_dims_dict
# from disdrodb.standards import get_var_explanations_ARM


# --------------------


def convert_standards(file_list, verbose):
    
    # from disdrodb.data_encodings import get_ARM_to_l0_dtype_standards
    # from disdrodb.standards import get_var_explanations_ARM
    
    dict_ARM = get_ARM_LPM_dict()
    
    # dict_ARM =      {
    #                 'time': 'time', # Name for time in Norway campaing
    #                 'base_time': 'time', # Name for time in ARM Mobile Facility campaing
    #                 'time_offset': 'time_offset_OldName',
    #                 'precip_rate': 'rainfall_rate_32bit',
    #                 'qc_precip_rate': 'qc_precip_rate_OldName',
    #                 'weather_code': 'weather_code_synop_4680',
    #                 'qc_weather_code': 'qc_weather_code_OldName',
    #                 'equivalent_radar_reflectivity_ott': 'reflectivity_32bit',
    #                 'qc_equivalent_radar_reflectivity_ott': 'qc_equivalent_radar_reflectivity_ott_OldName',
    #                 'number_detected_particles': 'number_particles',
    #                 'qc_number_detected_particles': 'qc_number_detected_particles_OldName',
    #                 'mor_visibility': 'mor_visibility_OldName',
    #                 'qc_mor_visibility': 'qc_mor_visibility_OldName',
    #                 'snow_depth_intensity': 'snow_depth_intensity_OldName',
    #                 'qc_snow_depth_intensity': 'qc_snow_depth_intensity_OldName',
    #                 'laserband_amplitude': 'laser_amplitude',
    #                 'qc_laserband_amplitude': 'qc_laserband_amplitude_OldName',
    #                 'sensor_temperature': 'sensor_temperature',
    #                 'heating_current': 'sensor_heating_current',
    #                 'qc_heating_current': 'qc_heating_current_OldName',
    #                 'sensor_voltage': 'sensor_battery_voltage',
    #                 'qc_sensor_voltage': 'qc_sensor_voltage_OldName',
    #                 'class_size_width': 'class_size_width_OldName',
    #                 'fall_velocity_calculated': 'fall_velocity_calculated_OldName',
    #                 'raw_spectrum': 'raw_spectrum_OldName',
    #                 'liquid_water_content': 'liquid_water_content_OldName',
    #                 'equivalent_radar_reflectivity': 'equivalent_radar_reflectivity_OldName',
    #                 'intercept_parameter': 'intercept_parameter_OldName',
    #                 'slope_parameter': 'slope_parameter_OldName',
    #                 'median_volume_diameter': 'median_volume_diameter_OldName',
    #                 'liquid_water_distribution_mean': 'liquid_water_distribution_mean_OldName',
    #                 'number_density_drops': 'number_density_drops_OldName',
    #                 'diameter_min': 'diameter_min_OldName',
    #                 'diameter_max': 'diameter_max_OldName',
    #                 'moment1': 'moment1_OldName',
    #                 'moment2': 'moment2_OldName',
    #                 'moment3': 'moment3_OldName',
    #                 'moment4': 'moment4_OldName',
    #                 'moment5': 'moment5_OldName',
    #                 'moment6': 'moment6_OldName',
    #                 'lat': 'latitude',
    #                 'lon': 'longitude',
    #                 'alt': 'altitude',
                    
    #                 # ALASKA
    #                 'synop_4677_weather_code': 'weather_code_synop_4677',
    #                 'metar_4678_weather_code': 'weather_code_metar_4678',
    #                 'synop_4680_weather_code': 'weather_code_synop_4680',
    #                 }
    
    # Custom dictonary for the campaign defined in standards
    dict_campaign = create_standard_dict(file_list[0], dict_ARM, verbose)
    
    # Log
    msg = f"Converting station "
    if verbose:
        print(msg)
    # logger.info(msg) 
    
    for f in file_list:
        file_name = campaign_name + '_' + station_id + '_' + str(file_list.index(f))
        output_dir = processed_dir + '/L1/' + file_name + '.nc'
        ds = xr.open_dataset(f)
        
        # Match field between NetCDF and dictionary
        list_var_names = list(ds.keys())
        dict_var = {k: dict_campaign[k] for k in dict_campaign.keys() if k in list_var_names}
        
        # Dimension dict
        dict_dims = get_ARM_LPM_dims_dict()
        
        # Rename NetCDF variables
        try:
            ds = ds.rename(dict_var)
            # Rename dimension
            ds = ds.rename_dims(dict_dims)
            # Rename coordinates
            ds = ds.rename(dict_dims)
        
        except Exception as e:
            msg = f"Error in rename variable. The error is: \n {e}"
            raise RuntimeError(msg)
            # To implement when move fuction into another file, temporary solution for now
            # logger.error(msg)
            # raise RuntimeError(msg)
    
        
        
        
        # ds = ds.drop(data_vars_to_drop)
        
        ds.to_netcdf(output_dir, mode='w', format="NETCDF4")
        ds.close()
        # Log
        msg = f"{file_name} processed successfully"
        if verbose:
            print(msg)
        # logger.info(msg)
    # Log
    msg = f"Station processed successfully"
    if verbose:
        print(msg)
    # logger.info(msg) 
    
def compare_standard_keys(dict_campaing, ds_keys, verbose):
    '''Compare a list (NetCDF keys) and a dictionary (standard from a campaing keys) and rename it, if a key is missin into the dictionary, take the missing key and add the suffix _OldName.'''
    dict_standard = {}
    count_skipped_keys = 0
    
    # Loop the NetCDF list
    for ds_v in ds_keys:
        # Loop standard dictionary for every element in list
        for dict_k, dict_v in dict_campaing.items():
            # If found a match, change the value with the dictionary standard and insert into a new dictionary
            if dict_k == ds_v:
                dict_standard[dict_k] = dict_v
                break
            else:
                # If doesn't found a match, insert list value with suffix into a new dictionary
                dict_standard[ds_v] = ds_v + '_TO_CHECK_VALUE_INTO_DATA_ENCONDINGS________'
                
                # Testing purpose
                # dict_standard[ds_v] = 'to_drop'
                
            # I don't kwow how implement counter :D
            # count_skipped_keys += 1
    
    count_skipped_keys = 'Not implemented'
    # Log
    if count_skipped_keys != 0:
        msg = f"Cannot convert keys values: {count_skipped_keys} on {len(ds_keys)}"
        if verbose:
            print(msg)
        # logger.info(msg) 
    
    return dict_standard


def create_standard_dict(file_path, dict_campaign, verbose):
    '''Insert a NetCDF keys into a list and return a dictionary compared with a defined standard dictionary (from a campaign)'''
    # Insert NetCDF keys into a list
    ds_keys = []
    
    ds = xr.open_dataset(file_path)
    
    for k in ds.keys():
        ds_keys.append(k)
        
    ds.close()
    
    dict_checked = compare_standard_keys(dict_campaign, ds_keys, verbose)
    
    # Compare NetCDF and dictionary keys
    return dict_checked
    
    
# --------------------


raw_dir = "/SharedVM/Campagne/ARM/Raw/ALASKA"
processed_dir = "/SharedVM/Campagne/ARM/Processed/ALASKA"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/ARM_MOBILE_FACILITY"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/ARM_MOBILE_FACILITY"
force = True
verbose = True
debugging_mode = True 

raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

campaign_name = get_campaign_name(raw_dir)


list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

for station_id in list_stations_id:
    
    print(f"Parsing station: {station_id}")

    glob_pattern = os.path.join("data", station_id, "*.nc") # CUSTOMIZE THIS 
    file_list = get_file_list(raw_dir=raw_dir,
                              glob_pattern=glob_pattern, 
                              verbose=verbose, 
                              debugging_mode=debugging_mode)
    
    

    convert_standards(file_list, verbose)