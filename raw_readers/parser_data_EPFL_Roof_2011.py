#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 13:01:54 2021

@author: kimbo
"""

import os
os.chdir(os.getcwd() + os.sep + os.pardir)
import glob 
import shutil
import click
import time
import dask.array as da
import numpy as np 
import xarray as xr
import pandas as pd
import json


from disdrodb.io import check_folder_structure
from disdrodb.io import check_valid_varname
from disdrodb.io import check_L0_standards
from disdrodb.io import check_L1_standards
from disdrodb.io import get_attrs_standards
from disdrodb.io import get_L0_dtype_standards
from disdrodb.io import get_dtype_standards_all_object
from disdrodb.io import get_flags
from disdrodb.io import _write_to_parquet
from disdrodb.io import col_dtype_check


from disdrodb.io import get_raw_field_nbins
from disdrodb.io import get_L1_coords
from disdrodb.io import rechunk_L1_dataset
from disdrodb.io import get_L1_zarr_encodings_standards
from disdrodb.io import get_L1_nc_encodings_standards

from disdrodb.logger import log
from disdrodb.logger import close_log

from disdrodb.attributes_campaing import Sensor
from disdrodb.attributes_campaing import Campaign
from disdrodb.attributes_campaing import read_JSON


from disdrodb.L1 import L1_process

#-------------------------------------------------------------------------.
# Click implementation

@click.command(options_metavar='<options>')

@click.argument('raw_dir', type=click.Path(exists=True), metavar ='<raw_dir>')

@click.argument('processed_path', metavar ='<processed_path>')

@click.option('--L0_processing',    '--L0',     is_flag=True, show_default=True, default = False,   help = 'Process the campaign in L0_processing')
@click.option('--L1_processing',    '--L1',     is_flag=True, show_default=True, default = False,   help = "Process the campaign in L1_processing")
@click.option('--force',            '--f',      is_flag=True, show_default=True, default = False,   help = "Force ...")
@click.option('--verbose',          '--v',      is_flag=True, show_default=True, default = False,   help = "Verbose ...")
@click.option('--debug_on',         '--d',      is_flag=True, show_default=True, default = False,   help = "Debug ...")
@click.option('--lazy',             '--l',      is_flag=True, show_default=True, default = True,    help = "Lazy ...")
@click.option('--keep_zarr',        '--kz',     is_flag=True, show_default=True, default = False,   help = "Keep zarr ...")
@click.option('--dtype_check',        '--dc',     is_flag=True, show_default=True, default = False,   help = "Check if the data are in the standars (max lenght, data range) ...")


# raw_dir = "/SharedVM/Campagne/ltnas3/Raw/EPFL_Roof_2011"
# processed_path = '/SharedVM/Campagne/ltnas3/Processed/EPFL_Roof_2011'
# L0_processing = False
# L1_processing = True
# force = True
# verbose = True
# debug_on = True
# lazy = True
# keep_zarr = False
# dtype_check = False



#-------------------------------------------------------------------------.


def main(raw_dir, processed_path, L0_processing, L1_processing, force, verbose, debug_on, lazy, keep_zarr, dtype_check):
    '''
    Script description
    
    <raw_dir>           : Raw file location of the campaign (example: <...>/Raw/<campaign name>, /ltenas3/0_Data/ParsivelDB/Raw/Ticino_2018)
    <processed_path>    : Processed file path output of the campaign (example: <...>/Processed/<campaign name>, /ltenas3/0_Data/ParsivelDB/Processed/Ticino_2018)
    
    
    '''
    
    #-------------------------------------------------------------------------.
    # Whether to use pandas or dask 
    if lazy: 
        import dask.dataframe as dd
    else: 
        import pandas as dd


    # ### Define attributes 
    attrs ={}
    
    ##------------------------------------------------------.   
    # Check the campaign path
    if os.path.isdir(raw_dir):
        pass
        #Path check in click, something else doesn't work
    else:
        print('Something wrong with path, quit script')
        raise SystemExit
        
    
    campaign_name = os.path.basename(processed_path)
    
    ##------------------------------------------------------.   
    # Check processed folder
    check_folder_structure(raw_dir, campaign_name, processed_path)
    
    campaign_name = os.path.basename(processed_path)
    
    ##------------------------------------------------------.   
    # Start log
    logger = log(processed_path, campaign_name)
    
    print('### Script start ###')
    logger.info('### Script start ###')
    
    ##------------------------------------------------------.   
    # Get campaign info and popolate device list
    
    json_flag = False
    
    # File path
    json_path = "/SharedVM/Campagne/ltnas3/Raw/EPFL_Roof_2011/EPFL_Roof_2011.json"
    
    campaign, device_list_info, device_list = read_JSON(json_path, processed_path, raw_dir, verbose)
    
    sensor_name = device_list_info[0].sensor_name
    
    ##------------------------------------------------------.   
    # Process all devices
    
    all_files = len(glob.glob(os.path.join(raw_dir, 'data', "**/*")))
    list_skipped_files = []
    
    msg = f'{all_files} files to process in {raw_dir}'
    if verbose:
        print(msg)
    logger.info(msg)
    
    for device in device_list_info:
        device_path = device.path
        
        ###############################
        #### Perform L0 processing ####
        ###############################
        
        # for device in 
        if L0_processing: 
            #----------------------------------------------------------------.
            if debug_on:
                print()
                print(' ***** Debug mode ON ***** ')
                print()
                rows_processed = 0
                
            if not lazy:
                print()
                print(' ***** Lazy mode OFF ***** ')
                print()
            
            t_i = time.time() 
            msg = f"L0 processing of device {device.disdrodb_id} started"
            if verbose:
                print(msg)
            logger.info(msg)
            
            file_list = sorted(glob.glob(os.path.join(device_path,"**/*.dat*"), recursive = True))
            
            if debug_on: 
                file_list = file_list[0:10]
                pass
            
            #----------------------------------------------------------------.
            # - Define raw data headers
            

            raw_data_columns = ['time', 
          						'id', 
          						'datalogger_temperature', 
          						'datalogger_voltage', 
          						'unknow', 
          						'rain_accumulated_32bit', 
          						'weather_code_SYNOP_4680',
          						'weather_code_SYNOP_4677',
          						'reflectivity_16bit',
          						'mor_visibility',
                                'laser_amplitude',
                                'n_particles',
                                'sensor_temperature',
                                'All_nan',
                                'sensor_heating_current',
                                'All_0',
                                'unknow2',
                                'Debug_data',
                                'FieldN',
                                'FieldV',
                                'RawData',
                                'End_line'
          						]
            
            
            # time_col = ['time']
            
            check_valid_varname(raw_data_columns)
            
            
            msg = f"{len(file_list)} files to process for {device.disdrodb_id}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##------------------------------------------------------.
            # Define reader options 
            reader_kwargs = {}
            reader_kwargs["engine"] = 'python'
            # - Replace custom NA with standard flags 
            reader_kwargs['na_values'] = ['', 'error', 'NA', 'na', '-.-']
            # Define time column
            # reader_kwargs['parse_dates'] = time_col
            reader_kwargs["blocksize"] = None
            reader_kwargs['header'] = None
            reader_kwargs['encoding'] = 'latin-1'  # Important for this campaign
            reader_kwargs['assume_missing'] = True
            
            ##------------------------------------------------------.
            # Loop over all files
            
            list_df = []
            for filename in file_list:
                
                ##------------------------------------------------------.      
                # All object for now
                dtype_dict = get_dtype_standards_all_object()
                
                try:
                    
                    df = dd.read_csv(filename,
                                    names = raw_data_columns,
                                    dtype=dtype_dict,
                                    **reader_kwargs
                                    )
                    
                    # Check if file empty
                    if len(df.index) == 0:
                        msg = f"{filename} is empty and has been skipped."
                        logger.warning(msg)
                        if verbose: 
                            print(msg)
                        list_skipped_files.append(msg)
                        continue
                        
                    # Remove unsused columns
                    df = df.drop(columns = ['All_nan','Debug_data', 'All_0', 'End_line'])
                    
                    #-------------------------------------------------------------------------
                    ### Keep only clean data 
                    # Drop rows with NA
                    df = df.dropna()
                    
                    ##------------------------------------------------------.
                    # Cast dataframe to dtypes
                    # Determine dtype based on standards 
                    dtype_dict = get_L0_dtype_standards()
    
                    for col in df.columns:
                        try:
                            df[col] = df[col].astype(dtype_dict[col])
                        except KeyError:
                            # If column dtype is not into L0_dtype_standards, assign object
                            df[col] = df[col].astype('object')
                            pass
                        except ValueError: # Work only on df.compute()
                            # cannot convert float NaN to integer, assign object
                            msg = f"{filename} on {col} cannot convert float NaN to integer, so cast to object"
                            logger.warning(msg)
                            if verbose: 
                                print(msg)
                            list_skipped_files.append(msg)
                            df[col] = df[col].astype('object')
                            pass
                    
                    
                        
                    ##------------------------------------------------------.
                    # Check dtype
                    if dtype_check:
                        col_dtype_check(df, filename, verbose)
    
                    # - Append to the list of dataframe 
                    list_df.append(df)
                    
                    msg = f'{filename} processed successfully'
                    if debug_on:
                        print(msg)
                    logger.debug(f'{filename} processed successfully')
                    
                except (Exception, ValueError) as e:
                  msg = f"{filename} has been skipped. The error is {e}"
                  logger.warning(msg)
                  if verbose: 
                      print(msg)
                  list_skipped_files.append(msg)
            
            msg = f"{len(list_skipped_files)} files on {len(file_list)} for {device.disdrodb_id} has been skipped."
            if verbose:      
                print(msg)
            logger.info('---')
            logger.info(msg)
            logger.info('---')
                
            ##------------------------------------------------------.
            # Concatenate the dataframe 
            msg = f"Start concat dataframes for device {device.disdrodb_id}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            try:
                df = dd.concat(list_df, axis=0, ignore_index = True)
                
                if debug_on:
                    print(f' ***** Proccessd rows for {device.disdrodb_id} are {len(df)} before time duplicates ***** ')
                
                # Drop duplicated values 
                df = df.drop_duplicates(subset="time")
                # Sort by increasing time 
                df = df.sort_values(by="time")
                
                if debug_on:
                    print(f' ***** Proccessd rows for {device.disdrodb_id} are {len(df)} after time duplicates ***** ') 
                
            except (AttributeError, TypeError) as e:
                msg = f"Can not create concat data files. Error: {e}"
                logger.exception(msg)
                raise ValueError(msg)
            
            msg = f"Finish concat dataframes for device {device.disdrodb_id}"
            if verbose:
                print(msg)
            logger.info(msg)
                
            ##------------------------------------------------------.                                   
            # Write to Parquet 
            msg = f"Starting conversion to parquet file for device {device.disdrodb_id}"
            if verbose:
                print(msg)
            logger.info(msg)
            # Path to device folder
            path = os.path.join(processed_path + '/' + os.path.basename(device.path) + '/L0/')
            # Write to Parquet 
            _write_to_parquet(df, path, campaign_name, force)
            
            msg = f"Finish conversion to parquet file for device {device.disdrodb_id}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##------------------------------------------------------.   
            # Check Parquet standards 
            check_L0_standards(df)
            
            ##------------------------------------------------------.   
            t_f = time.time() - t_i
            msg = "L0 processing of {} ended in {:.2f}s".format(device.disdrodb_id, t_f)
            if verbose:
                print(msg)
            logger.info(msg)
            
            if debug_on:
                print(f' ***** {len(df.index)} on {rows_processed} processed rows are saved ***** ')
            ##------------------------------------------------------.   
            # Delete temp variables
            del df
            del list_df
        
        #-------------------------------------------------------------------------.
        ###############################
        #### Perform L1 processing ####
        ###############################
        if L1_processing: 
            if debug_on:
                print()
                print(' ***** Debug mode ON ***** ')
                print()
                
            if not lazy:
                print()
                print(' ***** Lazy mode OFF ***** ')
                print()
                
        ##-----------------------------------------------------------
            t_i = time.time()  
            msg =f"L1 processing of device {device.disdrodb_id} started"
            if verbose:
                print(msg)
            logger.info(msg)
            
            json_flag = True
            
            L1_process(verbose, processed_path, campaign_name, L0_processing, lazy, debug_on, sensor_name, attrs, keep_zarr, device_list, device, json_flag)
            
            t_f = time.time() - t_i
            msg = "L1 processing of device {} ended in {:.2f}s".format(device.disdrodb_id, t_f)
            if verbose:
                print(msg)
            logger.info(msg)
            
        #-------------------------------------------------------------------------.
    
    msg = f'Total skipped files: {len(list_skipped_files)} on {all_files} in L0 process'
    if verbose:
        print(msg)
    logger.info('---')
    logger.info(msg)
    logger.info('---')
    
    msg = '### Script finish ###'
    print(msg)
    logger.info(msg)
    
    close_log()
    
    
if __name__ == '__main__':
    main() # when using click 
    # main(raw_dir, processed_path, L0_processing, L1_processing, force, verbose, debug_on, lazy, keep_zarr, dtype_check)
