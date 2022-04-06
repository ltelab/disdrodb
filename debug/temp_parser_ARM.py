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
    
    dict_ARM = get_ARM_LPM_dict()
    
    # Custom dictonary for the campaign defined in standards
    dict_campaign = create_standard_dict(file_list[0], dict_ARM, verbose)
    
    # Log
    msg = f"Converting station {station_id}"
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
        list_coords_names = list(ds.indexes)
        temp_dict_dims = get_ARM_LPM_dims_dict()
        dict_dims = {k: temp_dict_dims[k] for k in temp_dict_dims if k in list_coords_names}
        
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
    msg = f"Station {station_id} processed successfully"
    if verbose:
        print(msg)
    # logger.info(msg) 
    
def compare_standard_keys(dict_campaing, ds_keys, verbose):
    '''Compare a list (NetCDF keys) and a dictionary (standard from a campaing keys) and rename it, if a key is missin into the dictionary, take the missing key and add the suffix _OldName.'''
    dict_standard = {}
    # Initial list skipped keys
    list_skipped_keys = []

    # dict_standard = {k: dict_campaing[k] for k in dict_campaing if k in ds_keys}
    
    # Loop keys
    for ds_v in ds_keys:
        try:
            # Check if key into dict_campaing
            dict_standard[ds_v] = dict_campaing[ds_v]
        except KeyError:
            # If not present, give non standard name and add to list_skipped_keys
            dict_standard[ds_v] = ds_v + '_________TO_CHECK_VALUE_INTO_DATA_ENCONDINGS'
            list_skipped_keys.append(ds_v)
            pass
                
    # Log
    if list_skipped_keys:
        msg = f"Cannot convert keys values: {len(list_skipped_keys)} on {len(ds_keys)} \n Missing keys: {list_skipped_keys}"
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
    
    # Compare NetCDF and dictionary keys
    dict_checked = compare_standard_keys(dict_campaign, ds_keys, verbose)
    
    return dict_checked

def reformat_ARM_files(file_list, processed_dir, attrs):
    '''
    file_list:      List of NetCDF's path with the same ID
    processed_dir:  Save location for the renamed NetCDF
    dict_name:      Dictionary for rename NetCDF's variables (key: old name -> value: new name)
    attrs:          Info about campaing
    '''
    
    from disdrodb.L1_proc import get_L1_coords
    
    dict_ARM = get_ARM_LPM_dict()
    # Custom dictonary for the campaign defined in standards
    dict_campaign = create_standard_dict(file_list[0], dict_ARM, verbose)
    
    # Open netCDFs
    ds = xr.open_mfdataset(file_list)
    
    # Get coords
    coords = get_L1_coords(attrs['sensor_name'])
    
    # Assign coords and attrs
    coords["crs"] = attrs["crs"]
    coords["altitude"] = attrs["altitude"]
    
    # Match field between NetCDF and dictionary
    list_var_names = list(ds.keys())
    dict_var = {k: dict_campaign[k] for k in dict_campaign.keys() if k in list_var_names}
    
    # Dimension dict
    list_coords_names = list(ds.indexes)
    temp_dict_dims = get_ARM_LPM_dims_dict()
    dict_dims = {k: temp_dict_dims[k] for k in temp_dict_dims if k in list_coords_names}
    
    # Rename NetCDF variables
    try:
        ds = ds.rename(dict_var)
        # Rename dimension
        ds = ds.rename_dims(dict_dims)
        # Rename coordinates
        ds = ds.rename(dict_dims)
        # Assign coords
        ds = ds.assign_coords(coords)
        ds.attrs = attrs
    
    except Exception as e:
        msg = f"Error in rename variable. The error is: \n {e}"
        raise RuntimeError(msg)
        # To implement when move fuction into another file, temporary solution for now
        # logger.error(msg)
        # raise RuntimeError(msg)

    # data_vars_to_drop = []
    # ds = ds.drop(data_vars_to_drop)
    
    # Close NetCDF
    ds.close()
        
    return ds
    
    
# --------------------

# raw_dir = "/SharedVM/Campagne/ARM/Raw/ARGENTINA"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/ARGENTINA"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/ANTARTICA"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/ANTARTICA"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/ALASKA"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/ALASKA"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/ARM_MOBILE_FACILITY"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/ARM_MOBILE_FACILITY"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/BRAZIL"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/BRAZIL"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/MAR_SUPPLEMENTAL_FACILITY"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/MAR_SUPPLEMENTAL_FACILITY"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/NORWAY"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/NORWAY"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/PORTUGAL"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/PORTUGAL"
# raw_dir = "/SharedVM/Campagne/ARM/Raw/SOUTHERN_GREAT_PLAINS"
# processed_dir = "/SharedVM/Campagne/ARM/Processed/SOUTHERN_GREAT_PLAINS"
raw_dir = "/SharedVM/Campagne/ARM/Raw/SOUTHWEST_PACFIC_OCEAN"
processed_dir = "/SharedVM/Campagne/ARM/Processed/SOUTHWEST_PACFIC_OCEAN"

force = True
verbose = True
debugging_mode = True 

raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

campaign_name = get_campaign_name(raw_dir)

# Create directory structure
create_directory_structure(raw_dir, processed_dir)


list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

for station_id in list_stations_id:
    
    # Metadata 
    from disdrodb.metadata import read_metadata
    from disdrodb.check_standards import check_sensor_name
    
    # Retrieve metadata
    attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)

    # Retrieve sensor name
    sensor_name = attrs['sensor_name']
    check_sensor_name(sensor_name)
    
    print(f"Parsing station: {station_id}")

    glob_pattern = os.path.join("data", station_id, "*.cdf") # CUSTOMIZE THIS 
    file_list = get_file_list(raw_dir=raw_dir,
                              glob_pattern=glob_pattern, 
                              verbose=verbose, 
                              debugging_mode=debugging_mode)
    
    

    # convert_standards(file_list, verbose)
    reformat_ARM_files(file_list, processed_dir, attrs)