#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:30:51 2022

@author: ghiggi
"""
import os
import time
import glob 
import shutil
# Directory 
from disdrodb.io import check_directories
from disdrodb.io import get_campaign_name
from disdrodb.io import create_directory_structure

# Metadata 
from disdrodb.metadata import read_metadata
from disdrodb.check_standards import check_sensor_name

# L0A_processing
from disdrodb.check_standards import check_L0_standards
from disdrodb.L0_proc import get_file_list
from disdrodb.L0_proc import read_L0_raw_file_list
from disdrodb.L0_proc import write_df_to_parquet

# L0B_processing
from disdrodb.L1_proc import create_L1_dataset_from_L0
from disdrodb.L1_proc import write_L1_to_netcdf
from disdrodb.L1_proc import create_L1_summary_statistics

# Logger 
from disdrodb.logger import create_L0_logger
from disdrodb.logger import close_logger
from disdrodb.logger import log_info, log_warning
from disdrodb.io import get_L0A_dir, get_L0A_fpath, get_L0B_fpath
from disdrodb.io import read_L0A_dataframe   

#------------------------------------------------------------------.
# Consistency choice
# - Better name for raw_data_glob_pattern arg
# TODO: 
# - Add verbose and logs to disdrodb.io function !!!

#------------------------------------------------------------------.
def run_L0(
        raw_data_glob_pattern,
        column_names,
        reader_kwargs,
        df_sanitizer_fun, 
        raw_dir,
        processed_dir, 
        L0A_processing=True,
        L0B_processing=True,
        keep_L0A=True,
        force=False,
        verbose=False,
        debugging_mode=False,
        lazy=True,
        single_netcdf=True,
        ):
  
            
    t_i_script = time.time()
    #-------------------------------------------------------------------------.    
    # Initial directory checks 
    raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)
    
    # Retrieve campaign name 
    campaign_name = get_campaign_name(raw_dir)

    #-------------------------------------------------------------------------. 
    # Define logging settings
    logger = create_L0_logger(processed_dir, campaign_name)
            
    #-------------------------------------------------------------------------. 
    # Create directory structure 
    create_directory_structure(raw_dir, processed_dir)
                   
    #-------------------------------------------------------------------------. 
    #### Loop over station_id directory and process the files 
    list_stations_id = os.listdir(os.path.join(raw_dir, "data"))
    
    # station_id = list_stations_id[0]  
    for station_id in list_stations_id:
        #---------------------------------------------------------------------. 
        logger.info(f' - Processing of station_id {station_id} has started')
        #---------------------------------------------------------------------. 
        # Retrieve metadata 
        attrs = read_metadata(raw_dir=raw_dir,
                              station_id=station_id)
		# Retrieve sensor name
        sensor_name = attrs['sensor_name']
        check_sensor_name(sensor_name)
        
        #---------------------------------------------------------------------. 
        ######################## 
        #### L0A processing ####
        ######################## 
        if L0A_processing: 
            # Start L0 processing 
            t_i_station = time.time() 
            msg = " - L0A processing of station_id {} has started.".format(station_id)
            if verbose:
                print(msg)
            logger.info(msg)
            
            #-----------------------------------------------------------------.           
            #### - List files to process 
            glob_pattern = os.path.join("data", station_id, raw_data_glob_pattern)
            file_list = get_file_list(raw_dir=raw_dir,
                                      glob_pattern=glob_pattern, 
                                      verbose=verbose, 
                                      debugging_mode=debugging_mode)
            
            #-----------------------------------------------------------------. 
            #### - If single_netcdf = True, ensure loop over all files only once   
            if single_netcdf: 
                 file_list = [file_list]
            
            #-----------------------------------------------------------------.
            #### - Loop over all files 
            # - It loops only once if single_netcdf=True
            for filepath in file_list:
                ##------------------------------------------------------.
                # Define file suffix
                if single_netcdf: 
                    file_suffix = ""
                else: 
                    # Get file name without file extensions 
                    t_i_file = time.time() 
                    file_suffix = os.path.basename(filepath).split(".")[0]  
                    logger.info(f"L0A processing of raw file {file_suffix} has started.")
                
                ##------------------------------------------------------.
                #### - Read all raw data files into a dataframe  
                df = read_L0_raw_file_list(file_list=filepath,
                                           column_names=column_names,
                                           reader_kwargs=reader_kwargs,
                                           df_sanitizer_fun=df_sanitizer_fun,
                                           lazy=lazy,
                                           sensor_name=sensor_name,
                                           verbose=verbose)
            
                ##------------------------------------------------------.                                   
                #### - Write to Parquet                
                fpath = get_L0A_fpath(processed_dir, station_id, suffix=file_suffix)
                write_df_to_parquet(df=df,
                                    fpath=fpath,  
                                    force = force,
                                    verbose = verbose)
                
                ##------------------------------------------------------. 
                #### - Check L0 file respects the DISDRODB standards         
                check_L0_standards(fpath=fpath, 
                                   sensor_name=sensor_name, 
                                   verbose=verbose)
                
                ##------------------------------------------------------.
                # Delete temp variables
                del df
                
                ##------------------------------------------------------.
                if not single_netcdf:
                    # End L0 processing for a single raw file
                    t_f = time.time() - t_i_file
                    msg = " - L0A processing of {} ended in {:.2f}s".format(file_suffix, t_f)
                    log_info(logger, msg, verbose)
                    
            ##------------------------------------------------------. 
            # End L0 processing for the station
            t_f = time.time() - t_i_station
            msg = " - L0A processing of station_id {} ended in {:.2f}s".format(station_id, t_f)
            log_info(logger, msg, verbose)
                 
        #------------------------------------------------------------------.
        ########################
        #### L0B processing ####
        ########################
        if L0B_processing: 
            # Start L1 processing 
            t_i = time.time() 
            msg = " - L0B processing of station_id {} has started.".format(station_id)
            if verbose:
                print(msg)
            logger.info(msg)
            ##----------------------------------------------------------------.
            # Get station L0A directory  
            L0A_dir_path = get_L0A_dir(processed_dir, station_id)
            file_list = glob.glob(os.path.join(L0A_dir_path,"*.parquet"))
            n_files = len(file_list)
            if n_files == 0: 
               msg = f"No L0A Apache Parquet file is available in {L0A_dir_path}. Run L0A processing first."
               logger.error(msg)
               raise ValueError(msg)
            
            ##----------------------------------------------------------------.
            # Checks for single_netcdf=True
            if single_netcdf :
                # Enclose into a list to loop over only once 
                file_list = [file_list] 
                if n_files != 1: 
                    msg = "If single_netcdf=True, DISDRODB would typically expect only a single L0A Apache Parquet file in {L0A_dir_path}."
                    log_warning(logger, msg, verbose)
            
            ##----------------------------------------------------------------.
            # Loop over all files 
            for filepath in file_list:
                ##------------------------------------------------------.
                # Define file suffix
                if single_netcdf: 
                    file_suffix = ""
                else: 
                    # Get file name without file extensions 
                    t_i_file = time.time() 
                    file_suffix = os.path.basename(filepath).split(".")[0]  
                    logger.info(f"L0A processing of raw file {file_suffix} has started.")
                ##------------------------------------------------------.    
                # Read L0A dataframes                
                df = read_L0A_dataframe(filepath, 
                                        lazy=lazy,
                                        verbose=verbose,
                                        debugging_mode=debugging_mode)
                  
                #-----------------------------------------------------------------.
                #### - Create xarray Dataset
                ds = create_L1_dataset_from_L0(df=df, attrs=attrs, lazy=lazy, verbose=verbose)
                    
                #-----------------------------------------------------------------.
                #### - Write L0B netCDF4 dataset
                fpath = get_L0B_fpath(processed_dir, station_id, suffix=file_suffix)
                write_L1_to_netcdf(ds, fpath=fpath, sensor_name=sensor_name)
                
                #-----------------------------------------------------------------.
                if not single_netcdf:
                    # End L0B processing for a single L0A file
                    t_f = time.time() - t_i_file
                    msg = " - L0B processing of {} ended in {:.2f}s".format(file_suffix, t_f)
                    log_info(logger, msg, verbose)
                    
            #-----------------------------------------------------------------.
            #### - Compute L0B summary statics (if single_netcdf=True)
            if single_netcdf:
                create_L1_summary_statistics(ds, 
                                             processed_dir=processed_dir,
                                             station_id=station_id,
                                             sensor_name=sensor_name)
               
            #-----------------------------------------------------------------.
            # End L0B processing 
            t_f = time.time() - t_i
            msg = " - L0B processing of station_id {} ended in {:.2f}s".format(station_id, t_f)
            log_info(logger, msg, verbose)
 
            #-----------------------------------------------------------------.
        #---------------------------------------------------------------------.
    # Remove L0A directory if keep_L0A = False 
    if not keep_L0A: 
        shutil.rmtree(os.path.join(processed_dir, "L0A"))

    #-------------------------------------------------------------------------.
    # End of L0B processing for all stations
    t_f = time.time() - t_i_script
    msg = " - L0 processing of stations {} ended in {:.2f} minutes".format(list_stations_id, t_f/60)
    
    # Final logs 
    logger.info('---')    
    msg = '### Script finish ###'
    log_info(logger, msg, verbose)
    close_logger(logger)
    