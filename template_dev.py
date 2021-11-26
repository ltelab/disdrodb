#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:59:21 2021

@author: ghiggi
"""
import os
os.chdir("/home/kimbo/Documents/disdrodb")
import glob 
import shutil
import argparse # prefer click ... https://click.palletsprojects.com/en/8.0.x/
import click
import time
import dask.array as da
import numpy as np 
import xarray as xr 

from disdrodb.io import check_folder_structure
from disdrodb.io import check_valid_varname
from disdrodb.io import check_L0_standards
from disdrodb.io import check_L1_standards
from disdrodb.io import get_attrs_standards
from disdrodb.io import get_dtype_standards
from disdrodb.io import _write_to_parquet

from disdrodb.io import get_raw_field_nbins
from disdrodb.io import get_L1_coords
from disdrodb.io import rechunk_L1_dataset
from disdrodb.io import get_L1_zarr_encodings_standards
from disdrodb.io import get_L1_nc_encodings_standards

from disdrodb.logger import log

### TODO 4 Kimbo tomorrow

# To think eventually 
# --> Intermediate storage in Zarr: 74.32s
# --> Direct writing to netCDF: 61s

# loggerging -> File stuck open, it could be the global logger in io.py

## Infer Arrow schema from pandas
# import pyarrow as pa
# schema = pa.Schema.from_pandas(df)




#-------------------------------------------------------------------------.
# Click implementation

# @click.command(options_metavar='<options>')

# @click.argument('base_dir', type=click.Path(exists=True), metavar ='<base_dir>')

# @click.option('--l0_processing',    '--l0',     is_flag=True, show_default=True, default = False,   help = 'Process the campaign in l0_processing')
# @click.option('--l1_processing',    '--l1',     is_flag=True, show_default=True, default = False,   help = "Process the campaign in l1_processing")
# @click.option('--force',            '--f',      is_flag=True, show_default=True, default = False,   help = "Force ...")
# @click.option('--verbose',          '--v',      is_flag=True, show_default=True, default = False,   help = "Verbose ...")
# @click.option('--debug_on',         '--d',      is_flag=True, show_default=True, default = False,   help = "Debug ...")
# @click.option('--lazy',             '--l',      is_flag=True, show_default=True, default = True,    help = "Lazy ...")
# @click.option('--keep_zarr',        '--kz',     is_flag=True, show_default=True, default = False,   help = "Keep zarr ...")

base_dir = "/SharedVM/Campagne/Ticino_2018"
l0_processing = True
l1_processing = True
force = True
verbose = True
debug_on = True
lazy = False
keep_zarr = False


#-------------------------------------------------------------------------.


def main(base_dir, l0_processing, l1_processing, force, verbose, debug_on, lazy, keep_zarr):
    '''
    Script description
    
    <base_dir> : Campaign path
    
    
    '''
    
    #-------------------------------------------------------------------------.
    # Whether to use pandas or dask 
    if lazy: 
        import dask.dataframe as dd
    else: 
        import pandas as dd

    #-------------------------------------------------------------------------.
    # - Define instrument type 
    sensor_name = "Parsivel"
    
    # - Define attributes 
    attrs = get_attrs_standards()
    attrs['Title'] = 'aaa'
    attrs['lat'] = 1223
    attrs['lon'] = 1232
    attrs['crs'] = "WGS84"  # EPSG Code 
    attrs['sensor_name'] = sensor_name
    attrs['disdrodb_id'] = ''
    
 
    ###############################
    #### Perform L0 processing ####
    ###############################
    
    ##------------------------------------------------------.   
    # Check the campaign path
    if not os.path.exists(base_dir):
        raise FileExistsError('Not a directory, check the path')
    
    campaign_name = os.path.basename(base_dir)
    
    ##------------------------------------------------------.   
    # Start log
    logger = log(base_dir)
    
    print('### Script start ###')
    logger.info('### Script start ###')
    
    ##------------------------------------------------------.   
    # Check processed folder
    check_folder_structure(base_dir,campaign_name)
    
    campaign_name = os.path.basename(base_dir)
    
    ##------------------------------------------------------.   
    # Process all devices
    
    all_files = len(glob.glob(os.path.join(base_dir,"*")))
    list_skipped_files = []
    if verbose:
        print(f'{all_files} files to process in {base_dir}')
    logger.info(f'{all_files} files to process in {base_dir}')
    
    for device in glob.glob(os.path.join(base_dir,"data", "*")):
        
        # for device in 
        if l0_processing: 
            #----------------------------------------------------------------.
            t_i = time.time() 
            if verbose:
                print("L0 processing of {} started".format(attrs['disdrodb_id']))
            logger.info("L0 processing of {} started".format(attrs['disdrodb_id']))
            # - Retrieve filepaths of raw data files 
            # file_list = sorted(glob.glob(os.path.join(base_dir,"*")))
            # - With campaign path and all the stations files
            file_list = sorted(glob.glob(os.path.join(device,"**/*.dat*"), recursive = True))                
            
            if debug_on: 
                file_list = file_list[0:3]
            
            #----------------------------------------------------------------.
            # - Define raw data headers 
            raw_data_columns = ['id',
                                'latitude',
                                'longitude',
                                'time',
                                'temperature_sensor',
                                'datalogger_power',
                                'datalogger_sensor_status',
                                'rain_rate',
                                'acc_rain_amount',
                                'code_4680',
                                'code_4677',
                                'reflectivity_16bit',
                                'mor',
                                'amplitude',
                                'n_particles',
                                'heating_current',
                                'voltage',
                                'sensor_status',
                                'rain_amount_absolute',
                                'error_code',
                                'FieldN',
                                'FieldV',
                                'RawData',
                                'Unknow_column',
                                ]
            check_valid_varname(raw_data_columns)   
            
            ##------------------------------------------------------.      
            # Determine dtype based on standards 
            dtype_dict = get_dtype_standards()
            dtype_dict = {column: dtype_dict[column] for column in raw_data_columns}
            
            ##------------------------------------------------------.
            # Define reader options 
            reader_kwargs = {}
            reader_kwargs['compression'] = 'gzip'
            reader_kwargs['delimiter'] = ','
            reader_kwargs["on_bad_lines"] = 'skip'
            reader_kwargs["engine"] = 'python'
            if lazy:
                reader_kwargs["blocksize"] = None
            reader_kwargs["on_bad_lines"] = 'skip'
            
            ##------------------------------------------------------.
            # Loop over all files
            
            if verbose:
                print(f"{len(file_list)} files to process for {attrs['disdrodb_id']}")
            logger.info(f"{len(file_list)} files to process for {attrs['disdrodb_id']}")
            
            list_df = []
            for filename in file_list:
                try:
                    df = dd.read_csv(filename, 
                                     names = raw_data_columns, 
                                     dtype = dtype_dict,
                                     **reader_kwargs)
                    
                    # - Drop rows with all NaN
                    # ---> TODO: find a row with all NA 
                    # ---> TODO: remove rows with NA in specific columns 
                    df = df.dropna(how='all') 
                    
                    # - Replace custom NA with standard flags 
                    # TODO !!! 
                    
                    # - Append to the list of dataframe 
                    list_df.append(df)
                    
                    logger.debug(f'{filename} processed successfully')
                except (Exception, ValueError) as e:
                  msg = f"{filename} has been skipped. The error is {e}"
                  logger.warning(f'{filename} has been skipped')
                  if verbose: 
                      print(msg)
                  list_skipped_files.append(msg)
                 
            if verbose:      
                print(f"{len(list_skipped_files)} files on {len(file_list)} for {attrs['disdrodb_id']} has been skipped.")
            logger.info('---')
            logger.info(f"{len(list_skipped_files)} files on {len(file_list)} for {attrs['disdrodb_id']} has been skipped.")
            logger.info('---')
                
            
            ##------------------------------------------------------.
            # Concatenate the dataframe 
            if verbose:
                print(f"Start concat dataframes for {attrs['disdrodb_id']}")
            logger.info(f"Start concat dataframes for {attrs['disdrodb_id']}")
            
            try:
                df = dd.concat(list_df, axis=0, ignore_index = True)
            except (AttributeError, TypeError) as e:
                logger.exception(f"Can not create concat data files. Error: {e}")
                raise ValueError(f"Can not create concat data files. Error: {e}")
                
            if verbose:
                print(f"Finish concat dataframes for {attrs['disdrodb_id']}")
            logger.info(f"Finish concat dataframes for {attrs['disdrodb_id']}")
                
            ##------------------------------------------------------.                                   
            # Write to Parquet 
            if verbose:
                print(f"Starting conversion to parquet file for {attrs['disdrodb_id']}")
            logger.info(f"Starting conversion to parquet file for {attrs['disdrodb_id']}")
            # Path to device folder
            path = os.path.join(base_dir + "/processed/" + campaign_name + '/' + os.path.basename(device))
            # Write to Parquet 
            _write_to_parquet(df, path, campaign_name, force)
            if verbose:
                print(f"Finish conversion to parquet file for {attrs['disdrodb_id']}")
            logger.info(f"Finish conversion to parquet file for {attrs['disdrodb_id']}")
            
            ##------------------------------------------------------.   
            # Check Parquet standards 
            check_L0_standards(df)
            
            ##------------------------------------------------------.   
            t_f = time.time() - t_i
            if verbose:
                print("L0 processing of {} ended in {:.2f}s".format(attrs['disdrodb_id'], t_f))
            logger.info("L0 processing of {} ended in {:.2f}s".format(attrs['disdrodb_id'], t_f))
            
            ##------------------------------------------------------.   
            # Delete temp variables
            del df
            del list_df
        
        #-------------------------------------------------------------------------.
        ###############################
        #### Perform L1 processing ####
        ###############################
        if l1_processing: 
            ##-----------------------------------------------------------
            t_i = time.time()  
            if verbose:
                print("L1 processing of {} started".format(attrs['disdrodb_id']))
            logger.info("L1 processing of {} started".format(attrs['disdrodb_id']))
            
            ##-----------------------------------------------------------
            # Check the L0 df is available 
            # Path device folder parquet
            df_fpath = os.path.join(base_dir + "/processed/" + campaign_name + '/' + os.path.basename(device)) + '/' + campaign_name + '.parquet'
            if not l0_processing:
                if not os.path.exists(df_fpath):
                    logger.exception("Need to run L0 processing. The {df_fpath} file is not available.")
                    raise ValueError("Need to run L0 processing. The {df_fpath} file is not available.")
            if verbose:
                print(f'Found parquet file: {df_fpath}')
            logger.info(f'Found parquet file: {df_fpath}')
            
            ##-----------------------------------------------------------
            # Read raw data from parquet file 
            if verbose:
                print(f'Start reading: {df_fpath}')
            logger.info(f'Start reading: {df_fpath}')
            
            df = dd.read_parquet(df_fpath)
            
            if verbose:
                print(f'Finish reading: {df_fpath}')
            logger.info(f'Finish reading: {df_fpath}')
            
            ##-----------------------------------------------------------
            # Subset row sample to debug 
            if not lazy and debug_on:
                # df = df.iloc[0:100,:] # df.head(100) 
                df = df.iloc[0:,:]
                if verbose:
                    print(f'Debug = True and Lazy = False, then only the first 100 rows are read')
                logger.info(f'Debug = True and Lazy = False, then only the first 100 rows are read')
               
            ##-----------------------------------------------------------
            # Retrieve raw data matrix 
            if verbose:
                print(f"Retrieve raw data matrix for {attrs['disdrodb_id']}")
            logger.info(f"Retrieve raw data matrix for {attrs['disdrodb_id']}")
            
            dict_data = {}
            n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
            n_timesteps = df.shape[0]
            for key, n_bins in n_bins_dict.items(): 
                
                # Dask based 
                dd_series = df[key].astype(str).str.split(",")
                da_arr = da.stack(dd_series, axis=0)
                # Remove '' at the end 
                da_arr = da_arr[: , 0:n_bins_dict[key]]
                
                # Convert flag values to XXXX
                # dir(da_arr.str)
                # FieldN and FieldV --> -9.999, has floating number 
                
                if key == 'RawData':
                    da_arr = da_arr.astype(int)
                    da_arr = da_arr.reshape(n_timesteps, n_bins_dict['FieldN'], n_bins_dict['FieldV'])
                else:
                    da_arr = da_arr.astype(float)                
                
                dict_data[key] = da_arr
                           
                # Pandas/Numpy based 
                # np_arr_str =  df[key].values.astype(str)
                # list_arr_str = np.char.split(np_arr_str,",")
                # arr_str = np.stack(list_arr_str, axis=0) 
                # arr = arr_str[:, 0:n_bins]
                # arr = arr.astype(float)                
                # if key == 'RawData':
                #     arr = arr.reshape(n_timesteps, n_bins_dict['FieldN'], n_bins_dict['FieldV'])
                # dict_data[key] = arr
            
            if verbose:
                print(f"Finish retrieve raw data matrix for {attrs['disdrodb_id']}")
            logger.info(f"Finish retrieve raw data matrix for {attrs['disdrodb_id']}")
            
            ##-----------------------------------------------------------
            # Define data variables for xarray Dataset 
            data_vars = {"FieldN": (["time", "diameter_bin_center"], dict_data['FieldN']),
                          "FieldV": (["time", "velocity_bin_center"], dict_data['FieldV']),
                          "RawData": (["time", "diameter_bin_center", "velocity_bin_center"], dict_data['RawData']),
                        }
            
            # Define coordinates for xarray Dataset
            coords = get_L1_coords(sensor_name=sensor_name)
            coords['time'] = df['time'].values
            coords['lat'] = attrs['lat']
            coords['lon'] = attrs['lon']
            coords['crs'] = attrs['crs']
    
            ##-----------------------------------------------------------
            # Create xarray Dataset
            ds = xr.Dataset(data_vars = data_vars, 
                            coords = coords, 
                            attrs = attrs,
                            )
            
            ##-----------------------------------------------------------
            # Check L1 standards 
            check_L1_standards(ds)
            
            ##-----------------------------------------------------------    
            # Write to Zarr as intermediate storage 
            if keep_zarr:
                tmp_zarr_fpath = os.path.join(base_dir + "/processed/" + campaign_name + '/' + os.path.basename(device)) + '/' + campaign_name + '.zarr'
                ds = rechunk_L1_dataset(ds, sensor_name=sensor_name)
                zarr_encoding_dict = get_L1_zarr_encodings_standards(sensor_name=sensor_name)
                ds.to_zarr(tmp_zarr_fpath, encoding=zarr_encoding_dict)
            
            ##-----------------------------------------------------------  
            # Write L1 dataset to netCDF
            # Path for save into device folder
            path = os.path.join(base_dir + "/processed/" + campaign_name + '/' + os.path.basename(device))
            L1_nc_fpath = path + '/' + campaign_name + '.nc'
            ds = rechunk_L1_dataset(ds, sensor_name=sensor_name) # very important for fast writing !!!
            nc_encoding_dict = get_L1_nc_encodings_standards(sensor_name=sensor_name)
            
            if debug_on:
                ds.to_netcdf(L1_nc_fpath, engine="netcdf4")
            else:
                ds.to_netcdf(L1_nc_fpath, engine="netcdf4", encoding=nc_encoding_dict)
            
            
            ##-----------------------------------------------------------
            if verbose:
                t_f = time.time() - t_i
                print("L1 processing of {} ended in {:.2f}s".format(attrs['disdrodb_id'], t_f))
            logger.info("L1 processing of {} ended in {:.2f}s".format(attrs['disdrodb_id'], t_f))
        
        #-------------------------------------------------------------------------.
    
    if verbose:
        print(f'Total skipped files: {all_files} on {len(list_skipped_files)} in L0 process')
    logger.info('---')
    logger.info(f'Total skipped files: {all_files} on {len(list_skipped_files)} in L0 process')
    logger.info('---')
    
    print('### Script finish ###')
    logger.info('### Script finish ###')
 

if __name__ == '__main__':
    # main() # when using click 
    # main()
    main(base_dir, l0_processing, l1_processing, force, verbose, debug_on, lazy, keep_zarr)
    
    # Otherwise:     
    # parser = argparse.ArgumentParser(description='L0 and L1 data processing')
    # parser.add_argument('--base_dir', type=str)
    # parser.add_argument('--l0_processing', type=str, default='True')
    # parser.add_argument('--l1_processing', type=str, default='True')
    # parser.add_argument('--force', type=str, default='False')                    
    
    # l0_processing=True, l1_processing=True, force=False
    
    # args = parser.parse_args()
    # if args.force == 'True':
    #     force = True
    # else: 
    #     force = False
    # if args.l0_processing == 'True':
    #     l0_processing = True
    # else: 
    #     l0_processing = False 
    #  if args.l1_processing == 'True':
    #     l1_processing = True
    # else: 
    #     l1_processing = False   
        
    # main(base_dir = base_dir, 
    #      l0_processing=l0_processing, 
    #      l1_processing=l1_processing,
    #      force=force)
 
