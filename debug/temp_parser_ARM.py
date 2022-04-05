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
    
    print(f"Parsing station: {station_id}")

    glob_pattern = os.path.join("data", station_id, "*.cdf") # CUSTOMIZE THIS 
    file_list = get_file_list(raw_dir=raw_dir,
                              glob_pattern=glob_pattern, 
                              verbose=verbose, 
                              debugging_mode=debugging_mode)
    
    

    convert_standards(file_list, verbose)