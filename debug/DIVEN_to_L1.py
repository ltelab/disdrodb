#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:27:30 2022

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import os
import xarray as xr
import netCDF4

import logging
import time
import datetime

from disdrodb.data_encodings import get_DIVEN_dict
from disdrodb.data_encodings import get_ARM_LPM_dict


# Directory 
from disdrodb.io import check_directories
from disdrodb.io import get_campaign_name
from disdrodb.io import create_directory_structure

# Logger 
from disdrodb.logger import create_logger
from disdrodb.logger import close_logger

# Metadata 
from disdrodb.metadata import read_metadata
from disdrodb.check_standards import check_sensor_name

# L0_processing
from disdrodb.L0_proc import get_file_list

############## 

raw_dir = '/SharedVM/Campagne/DIVEN/Raw/CAIRNGORM'
processed_dir = '/SharedVM/Campagne/DIVEN/Processed/CAIRNGORM'
force = True
rename_netcdf = True
verbose = True
l0_processing = True
l1_processing = True
debugging_mode = False


### Function ###

def rename_variable_netcdf(file_list, processed_dir, dict_name):
    
    from datetime import date
    
    for f in file_list:
        
        # Counter
        x = 0

        ds = xr.open_dataset(f)
    
        # print(list(ds.keys()))
        # print(len(list(ds.keys())))
    
        # Match field between NetCDF and dictionary
        list_var_names = list(ds.keys())
    
        dict_var = {k: dict_name[k] for k in dict_name.keys() & set(list_var_names)}
    
        ds = ds.rename(dict_var)
    
        # print(list(ds.keys()))
        # print(len(list(ds.keys())))
        
        path = processed_dir + '/L1/' + station_id + '/' + campaign_name + '_' + station_id + '_' + str(date.today()) + '_nr_' + str(x) +'.nc'

        ds.to_netcdf(path=path)
    
        ds.close()
        
        x += 1


##------------------------------------------------------------------------.
#### - Define glob pattern to search data files in raw_dir/data/<station_id>
raw_data_glob_pattern= "*.nc*"

####----------------------------------------------------------------------.
####################
#### FIXED CODE ####
####################
# -------------------------------------------------------------------------.
# Initial directory checks
raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)

# Retrieve campaign name
campaign_name = get_campaign_name(raw_dir)

# -------------------------------------------------------------------------.
# Define logging settings
create_logger(processed_dir, "parser_" + campaign_name)
# Retrieve logger
logger = logging.getLogger(campaign_name)
logger.info("### Script started ###")

# -------------------------------------------------------------------------.
# Create directory structure
create_directory_structure(raw_dir, processed_dir)


#### Loop over station_id directory and process the files
list_stations_id = os.listdir(os.path.join(raw_dir, "data"))

# station_id = list_stations_id[1]
for station_id in list_stations_id:
    # ---------------------------------------------------------------------.
    logger.info(f" - Processing of station_id {station_id} has started")
    # ---------------------------------------------------------------------.
    # Retrieve metadata
    attrs = read_metadata(raw_dir=raw_dir, station_id=station_id)
    
    # Retrieve sensor name
    sensor_name = attrs['sensor_name']
    check_sensor_name(sensor_name)
    
    # ---------------------------------------------------------------------.
    #######################
    #### Rename NetCDF ####
    #######################
    if rename_netcdf:
        
        # Start rename processing
        t_i = time.time()
        msg = " - L0 processing of station_id {} has started.".format(station_id)
        if verbose:
            print(msg)
        logger.info(msg)

        # -----------------------------------------------------------------.
        #### - List files to process
        glob_pattern = os.path.join("data", station_id, raw_data_glob_pattern)
        file_list = get_file_list(
            raw_dir=raw_dir,
            glob_pattern=glob_pattern,
            verbose=verbose,
            debugging_mode=debugging_mode,
        )
        
        # Dictionary name
        dict_name = get_DIVEN_dict()
        
        rename_variable_netcdf(file_list, processed_dir, dict_name)
        

    # ---------------------------------------------------------------------.
    #######################
    #### L0 processing ####
    #######################
    if l0_processing:
        
        msg = (' - Not needed L0 processing, so skipped')
        
        if verbose:
            print(msg)
            print(" --------------------------------------------------")
        logger.info(msg)

    # ---------------------------------------------------------------------.
    #######################
    #### L1 processing ####
    #######################
    if l1_processing:
        
        print(' - Not needed L1 processing, so skipped')
        
        if verbose:
            print(msg)
            print(" --------------------------------------------------")
        logger.info(msg)
    # ---------------------------------------------------------------------.
# -------------------------------------------------------------------------.
if verbose:
    print(msg)
logger.info("---")
logger.info(msg)
logger.info("---")

msg = "\n   ### Script finish ###"
print(msg)
logger.info(msg)

close_logger(logger)


#################################

dict_DIVEN = get_DIVEN_dict()

file_path = '/SharedVM/Campagne/DIVEN/Raw/cairngorm/data/11/ncas-disdrometer-11_cairngorm_20170210_precipitation_v1.0.nc'

ds = xr.open_dataset(file_path)

print(list(ds.keys()))
print(len(list(ds.keys())))


# Match field between NetCDF and dictionary
list_var_names = list(ds.keys())

dict_var = {k: dict_DIVEN[k] for k in dict_DIVEN.keys() & set(list_var_names)}

ds = ds.rename(dict_var)

print(list(ds.keys()))
print(len(list(ds.keys())))

from datetime import date
path = processed_dir + '/L1/' + station_id + '/' + campaign_name + '_' + station_id + '_' + str(date.today()) + '.nc'

ds.to_netcdf(path=path)

ds.close()



