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
from disdrodb.data_encodings import get_ARM_to_l0_dtype_standards
from disdrodb.standards import get_var_explanations_ARM


# --------------------


def convert_standards(file_list, verbose):
    
    from disdrodb.data_encodings import get_ARM_to_l0_dtype_standards
    from disdrodb.standards import get_var_explanations_ARM
    
    dict_ARM = get_ARM_to_l0_dtype_standards()
    
    # Log
    msg = f"Converting station "
    if verbose:
        print(msg)
    # logger.info(msg) 
    
    for f in file_list:
        file_name = campaign_name + '_' + station_id + '_' + str(file_list.index(f))
        output_dir = processed_dir + '/L1/' + file_name
        ds = xr.open_dataset(f)
        ds = ds.rename(dict_ARM)
        ds.to_netcdf(output_dir, mode='w', format="NETCDF4")
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
    
    
# --------------------


raw_dir = "/SharedVM/Campagne/ARM/Raw/NORWAY" # Must end with campaign_name upper case
processed_dir = "/SharedVM/Campagne/ARM/Processed/NORWAY"  # Must end with campaign_name upper case
force = True
verbose = True
debugging_mode = True 

raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

campaign_name = get_campaign_name(raw_dir)


list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

station_id = list_stations_id[0]

glob_pattern = os.path.join("data", station_id, "*.cdf") # CUSTOMIZE THIS 
file_list = get_file_list(raw_dir=raw_dir,
                          glob_pattern=glob_pattern, 
                          verbose=verbose, 
                          debugging_mode=debugging_mode)

convert_standards(file_list, verbose)