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
import click

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

# IO
from disdrodb.io import get_L0_fpath
from disdrodb.io import get_L1_netcdf_fpath

# L1_processing
from disdrodb.L1_proc import write_L1_to_netcdf

############## 

# raw_dir = '/SharedVM/Campagne/DIVEN/Raw/CAPEL-DEWI'
# processed_dir = '/SharedVM/Campagne/DIVEN/Processed/CAPEL-DEWI'
# force = True
# rename_netcdf = True
# verbose = True
# l0_processing = True
# l1_processing = True
# debugging_mode = True

### Function ###

def rename_variable_netcdf(file_list, processed_dir, dict_name, attrs):
    '''
    file_list:      List of NetCDF's path with the same ID
    processed_dir:  Save location for the renamed NetCDF
    dict_name:      Dictionary for rename NetCDF's variables (key: old name -> value: new name)
    attrs:          Info about campaing
    '''
    
    from disdrodb.L1_proc import get_L1_coords
    
    # Open netCDFs
    ds = xr.open_mfdataset(file_list)
    
    # Get coords
    coords = get_L1_coords(attrs['sensor_name'])
        
    # Remove lat and long
    ds = ds.squeeze()
    
    # Rename diameter and fallspeed to diameter_bin_center and velocity_bin_center
    ds = ds.rename_dims({'diameter': 'diameter_bin_center', 'fallspeed': 'velocity_bin_center'})
    # ds = ds.drop(['diameter', 'fallspeed'])
    
    # Assign coords and attrs
    coords["crs"] = attrs["crs"]
    coords["altitude"] = attrs["altitude"]
    ds = ds.assign_coords(coords)
    ds.attrs = attrs
    
    # Drop useless data variables
    data_vars_to_drop = ['year',
                         'month',
                         'day',
                         'hour',
                         'minute',
                         'second',
                         'day_of_year',
                         'qc_flag',
                         'measurement_quality',
                         'present_weather_5m',
                         'hydrometeor_type_1m',
                         'hydrometeor_type_5m',
                         'max_hail_diameter']
    
    ds = ds.drop(data_vars_to_drop)
    
    # Match field between NetCDF and dictionary
    list_var_names = list(ds.keys())
    dict_var = {k: dict_name[k] for k in dict_name.keys() & set(list_var_names)}
    
    # Rename NetCDF variables
    try:
        ds = ds.rename(dict_var)
    
    except Exception as e:
        msg = f"Error in rename variable. The error is: \n {e}"
        raise RuntimeError(msg)
        # To implement when move fuction into another file, temporary solution for now
        # logger.error(msg)
        # raise RuntimeError(msg)
    
    # Close NetCDF
    ds.close()
        
    return ds

# -------------------------------------------------------------------------.
# CLIck Command Line Interface decorator
# @click.command()  # options_metavar='<options>'
# @click.argument('raw_dir', type=click.Path(exists=True), metavar='<raw_dir>')
# @click.argument('processed_dir', metavar='<processed_dir>')
# @click.option('-l0', '--l0_processing', type=bool, show_default=True, default=True, help="Perform L0 processing")
# @click.option('-l1', '--l1_processing', type=bool, show_default=True, default=True, help="Perform L1 processing")
# @click.option('-nc', '--write_netcdf', type=bool, show_default=True, default=True, help="Write L1 netCDF4")
# @click.option('-f', '--force', type=bool, show_default=True, default=False, help="Force overwriting")
# @click.option('-v', '--verbose', type=bool, show_default=True, default=False, help="Verbose")
# @click.option('-d', '--debugging_mode', type=bool, show_default=True, default=False, help="Switch to debugging mode")
# @click.option('-l', '--lazy', type=bool, show_default=True, default=True, help="Use dask if lazy=True")
# @click.option('-rn', '--rename_netcdf', type=bool, show_default=True, default=True, help="Rename NetCDF variables by a defined dictionary")
def main(raw_dir,
         processed_dir,
         l0_processing=True,
         l1_processing=True,
         write_netcdf=False,
         force=True,
         verbose=True,
         debugging_mode=False,
         lazy=True,
         rename_netcdf = True
         ):


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
    msg = "### Script started ###"
    if verbose:
        print("\n  " + msg + "\n")
    logger.info(msg)
    
    # -------------------------------------------------------------------------.
    # Create directory structure
    create_directory_structure(raw_dir, processed_dir)
    
    
    #### Loop over station_id directory and process the files
    list_stations_id = os.listdir(os.path.join(raw_dir, "data"))
    
    # station_id = list_stations_id[1]
    for station_id in list_stations_id:
        # ---------------------------------------------------------------------.
        msg = f" - Processing of station_id {station_id} has started"
        if verbose:
            print(msg)
        logger.info(msg)
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
            msg = " - Rename NetCDF of station_id {} has started.".format(station_id)
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
            
            ds = rename_variable_netcdf(file_list, processed_dir, dict_name, attrs)
            
            fpath = get_L1_netcdf_fpath(processed_dir, station_id)
            write_L1_to_netcdf(ds, fpath=fpath, sensor_name=sensor_name)
            
            # End L0 processing
            t_f = time.time() - t_i
            msg = " - Rename NetCDF processing of station_id {} ended in {:.2f}s".format(
                station_id, t_f
            )
            if verbose:
                print(msg)
            logger.info(msg)
            
            msg = (" --------------------------------------------------")
            if verbose:
                print(msg)
            logger.info(msg)
            
    
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
            
            msg = (' - Not needed L1 processing, so skipped')
            
            if verbose:
                print(msg)
                print(" --------------------------------------------------")
            logger.info(msg)
        # ---------------------------------------------------------------------.
    # -------------------------------------------------------------------------.
    
    msg = "### Script finish ###"
    print("\n  " + msg + "\n")
    logger.info(msg)
    
    close_logger(logger)

#################################

if __name__ == "__main__":
    # main()
    main(
        raw_dir = '/SharedVM/Campagne/DIVEN/Raw/CAPEL-DEWI',
        processed_dir = '/SharedVM/Campagne/DIVEN/Processed/CAPEL-DEWI',
        l0_processing=True,
        l1_processing=True,
        write_netcdf=False,
        force=True,
        verbose=True,
        debugging_mode=False,
        lazy=True,
        rename_netcdf = True)



