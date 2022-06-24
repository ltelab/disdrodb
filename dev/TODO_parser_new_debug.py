#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:26:27 2022

@author: ghiggi
"""
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
import os
import click
import logging
from disdrodb.check_standards import check_L0_column_names

 
raw_dir = "/ltenas3/0_Data/DISDRODB/Raw/EPFL/LOCARNO_2018"
processed_dir = "/tmp/DISDRODB/Processed/EPFL/LOCARNO_2018"
L0A_processing=True
L0B_processing=True
keep_L0B=True
force=True
verbose=True
debugging_mode=True
lazy=True
single_netcdf=True

## TO DEBUG: 
# create_L1_dataset_from_L0() still crash when lazy=True !!!

# -------------------------------------------------------------------------.
# CLIck Command Line Interface decorator
@click.command()  # options_metavar='<options>'
@click.argument('raw_dir', type=click.Path(exists=True), metavar='<raw_dir>')
@click.argument('processed_dir', metavar='<processed_dir>')
@click.option('-L0A', '--L0A_processing', type=bool, show_default=True, default=True, help="Perform L0A processing")
@click.option('-L0B', '--L0B_processing', type=bool, show_default=True, default=True, help="Perform L0B processing")
@click.option('-k', '--keep_L0B', type=bool, show_default=True, default=True, help="Whether to keep the L0A Parquet file.")
@click.option('-f', '--force', type=bool, show_default=True, default=False, help="Force overwriting")
@click.option('-v', '--verbose', type=bool, show_default=True, default=False, help="Verbose")
@click.option('-d', '--debugging_mode', type=bool, show_default=True, default=False, help="Switch to debugging mode")
@click.option('-l', '--lazy', type=bool, show_default=True, default=True, help="Use dask if lazy=True")
@click.option('-s', '--single_netcdf', type=bool, show_default=True, default=True, help="Produce single netCDF.")
def main(raw_dir,
         processed_dir,
         L0A_processing=True,
         L0B_processing=True,
         keep_L0B=True,
         force=False,
         verbose=False,
         debugging_mode=False,
         lazy=True,
         single_netcdf = True, 
         ):
    """Script to process raw data to L0 and L1. \f
    
    Parameters
    ----------
    raw_dir : str
        Directory path of raw file for a specific campaign.
        The path should end with <campaign_name>.
        Example raw_dir: '<...>/disdrodb/data/raw/<campaign_name>'.
        The directory must have the following structure:
        - /data/<station_id>/<raw_files>
        - /metadata/<station_id>.json 
        For each <station_id> there must be a corresponding JSON file
        in the metadata subfolder.
    processed_dir : str
        Desired directory path for the processed L0 and L1 products. 
        The path should end with <campaign_name> and match the end of raw_dir.
        Example: '<...>/disdrodb/data/processed/<campaign_name>'.
    L0A_processing : bool
      Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
      The default is True.
    L0B_processing : bool
      Whether to launch processing to generate L0B netCDF4 file(s) from raw (or L0B) data. 
      The default is True.
    force : bool
        If True, overwrite existing data into destination directories. 
        If False, raise an error if there are already data into destination directories. 
        The default is False
    verbose : bool
        Whether to print detailed processing information into terminal. 
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        - For L0 processing, it processes just 3 raw data files.
        - For L1 processing, it takes a small subset of the Apache Parquet dataframe.
        The default is False.
    lazy : bool
        Whether to perform processing lazily with dask. 
        If lazy=True, it employed dask.array and dask.dataframe.
        If lazy=False, it employed pandas.DataFrame and numpy.array.
        The default is True.
    single_netcdf : bool
        Whether to concatenate all raw files into a single L0 netCDF file.
        If single_netcdf=True, all raw files will be saved into a single L0 netCDF file.
        If single_netcdf=False, each raw file will be converted into a single L0 netCDF file.
        The default is True.
    
    Additional information:
    - The campaign name must semantically match between:
       - The ends of raw_dir and processed_dir paths 
       - The attribute 'campaign' within the metadata JSON file. 
    - The campaign name are set to be UPPER CASE. 
       
    """
    ####----------------------------------------------------------------------.
    ###########################
    #### CUSTOMIZABLE CODE ####
    ###########################
    #### - Define raw data headers 
    # Notes
    # - In all files, the datalogger voltage hasn't the delimeter, 
    #   so need to be split to obtain datalogger_voltage and rainfall_rate_32bit 
    column_names = ['id',
                    'latitude',
                    'longitude',
                    'time',
                    'datalogger_temperature',
                    'datalogger_voltage',
                    'rainfall_rate_32bit',
                    'rainfall_accumulated_32bit',
                    'weather_code_synop_4680',
                    'weather_code_synop_4677',
                    'reflectivity_32bit',
                    'mor_visibility',
                    'laser_amplitude',  
                    'number_particles',
                    'sensor_temperature',
                    'sensor_heating_current',
                    'sensor_battery_voltage',
                    'sensor_status',
                    'rainfall_amount_absolute_32bit',
                    'error_code',
                    'raw_drop_concentration',
                    'raw_drop_average_velocity',
                    'raw_drop_number',
                    'datalogger_error'
                    ]
    
    # - Check name validity 
    check_L0_column_names(column_names)
    
    ##------------------------------------------------------------------------.
    #### - Define reader options 
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs['delimiter'] = ','
    
    # - Avoid first column to become df index 
    reader_kwargs["index_col"] = False  
    
    # - Define behaviour when encountering bad lines 
    reader_kwargs["on_bad_lines"] = 'skip'
    
    # - Define parser engine 
    #   - C engine is faster
    #   - Python engine is more feature-complete
    reader_kwargs["engine"] = 'python'
    
    # - Define on-the-fly decompression of on-disk data
    #   - Available: gzip, bz2, zip 
    reader_kwargs['compression'] = 'infer'  
    
    # - Strings to recognize as NA/NaN and replace with standard NA flags 
    #   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, 
    #                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’, 
    #                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
    reader_kwargs['na_values'] = ['na', '', 'error']
    
    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None # "50MB" 
    
    ##------------------------------------------------------------------------.
    #### - Define facultative dataframe sanitizer function for L0 processing
    # - Enable to deal with bad raw data files 
    # - Enable to standardize raw data files to L0 standards  (i.e. time to datetime)
    df_sanitizer_fun = None 
    def df_sanitizer_fun(df, lazy=False):
        # Import dask or pandas 
        if lazy: 
            import dask.dataframe as dd
        else: 
            import pandas as dd

        # - Drop datalogger columns 
        columns_to_drop = ['id', 'datalogger_temperature', 'datalogger_voltage', 'datalogger_error']
        df = df.drop(columns=columns_to_drop)
        
        # - Drop latitude and longitute (always the same)
        df = df.drop(columns=['latitude', 'longitude'])
        
        # - Convert time column to datetime 
        df['time'] = dd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
        
        return df  
    
    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    raw_data_glob_pattern = "*.dat*"   

    ####----------------------------------------------------------------------.
    #################### 
    #### FIXED CODE ####
    #################### 

    # run_L0_standardization(raw_dir=raw_dir,  
    #                        processed_dir=processed_dir
    #                        L0A_processing=L0A_processing,
    #                        L0B_processing=L0B_processing,
    #                        keep_L0B=keep_L0B,
    #                        force=force,
    #                        verbose=verbose,
    #                        debugging_mode=debugging_mode,
    #                        lazy=lazy,
    #                        single_netcdf=single_netcdf,
    #                        # Custom of the parser 
    #                        raw_data_glob_pattern = raw_data_glob_pattern, 
    #                        column_names=column_names,
    #                        reader_kwargs=reader_kwargs,
    #                        df_sanitizer_fun=df_sanitizer_fun,
    #                        )
    
    ### TO BE PLACED IN disdrodb.L0.proc_py.
    # def run_L0_standardization(
    #                        raw_data_glob_pattern,
    #                        column_names,
    #                        reader_kwargs,
    #                        df_sanitizer_fun, 
    #                        raw_dir, processed_dir, 
    #                        L0A_processing=True,
    #                        L0B_processing=True,
    #                        keep_L0B=True,
    #                        force=False,
    #                        verbose=False,
    #                        debugging_mode=False,
    #                        lazy=True,
    #                        single_netcdf = True,
    #                        )
    import time
    import glob 
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
    # TODO: REMOVE L0A directory if keep_L0A is FALSE 
    # TODO
    # TODO
    #-------------------------------------------------------------------------.
    # End of L0B processing for all stations
    t_f = time.time() - t_i_script
    msg = " - L0 processing of stations {} ended in {:.2f} minutes".format(list_stations_id, t_f/60)
    
    # Final logs 
    logger.info('---')    
    msg = '### Script finish ###'
    log_info(logger, msg, verbose)
    close_logger(logger)
    
 
if __name__ == '__main__':
    main()  
