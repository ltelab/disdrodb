#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:31:45 2021

@author: kimbo
"""

import os
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
# os.chdir("/home/kimbo/Documents/disdrodb")
import glob 
import shutil
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

from disdrodb.sensor import Sensor

### TODO 4 Kimbo tomorrow

# To think eventually 
# --> Intermediate storage in Zarr: 74.32s
# --> Direct writing to netCDF: 61s

## Infer Arrow schema from pandas
# import pyarrow as pa
# schema = pa.Schema.from_pandas(df)

# Error file busy on zarr, don't overwrite
# Error on lazy with L1

# TODO
# - Ensure not time duplicate !!!
# - Ensure time ordered !!!

# Make template work for Ticino, Payerne, 1 ARM Parsivel, 1 UK Diven, 1 Hymex


#-------------------------------------------------------------------------.
# Click implementation

# @click.command(options_metavar='<options>')

# @click.argument('raw_dir', type=click.Path(exists=True), metavar ='<raw_dir>')

# @click.argument('processed_path', metavar ='<processed_path>') #TODO

# @click.option('--L0_processing',    '--L0',     is_flag=True, show_default=True, default = False,   help = 'Process the campaign in L0_processing')
# @click.option('--L1_processing',    '--L1',     is_flag=True, show_default=True, default = False,   help = "Process the campaign in L1_processing")
# @click.option('--force',            '--f',      is_flag=True, show_default=True, default = False,   help = "Force ...")
# @click.option('--verbose',          '--v',      is_flag=True, show_default=True, default = False,   help = "Verbose ...")
# @click.option('--debug_on',         '--d',      is_flag=True, show_default=True, default = False,   help = "Debug ...")
# @click.option('--lazy',             '--l',      is_flag=True, show_default=True, default = True,    help = "Lazy ...")
# @click.option('--keep_zarr',        '--kz',     is_flag=True, show_default=True, default = False,   help = "Keep zarr ...")
# @click.option('--dtype_check',        '--dc',     is_flag=True, show_default=True, default = False,   help = "Check if the data are in the standars (max lenght, data range) ...")


raw_dir = "/SharedVM/Campagne/ltnas3/Raw/COMMON_2011"
processed_path = '/SharedVM/Campagne/ltnas3/Processed/COMMON_2011'
L0_processing = True
L1_processing = True
force = True
verbose = True
debug_on = True
lazy = True
keep_zarr = False
dtype_check = False



#-------------------------------------------------------------------------.


def main(raw_dir, processed_path, L0_processing, L1_processing, force, verbose, debug_on, lazy, keep_zarr, dtype_check):
    '''
    Script description
    
    <raw_dir>           : Raw file location of the campaign (example: <...>/Raw/<campaign name>, /ltenas3/0_Data/ParsivelDB/Raw/Ticino_2018)
    <processed_path>    : Processed file path output of the campaign (example: <...>/Processed/<campaign name>, /ltenas3/0_Data/ParsivelDB/Processed/Ticino_2018)
    
    
    '''
    #-------------------------------------------------------------------------.
    # Hard coded server path
    # processed_path = '/ltenas3/0_Data/ParsivelDB/Processed/Ticino_2018'
    
    #-------------------------------------------------------------------------.
    # Whether to use pandas or dask 
    if lazy: 
        import dask.dataframe as dd
    else: 
        import pandas as dd

    #-------------------------------------------------------------------------.
    # - Define instrument type 
    sensor_name = ""
    #-------------------------------------------------------------------------.
    ### Define attributes 
    attrs = get_attrs_standards()
    # - Description
    attrs['title'] = 'Common_2011'
    attrs['description'] = '' 
    attrs['institution'] = 'Laboratoire de Teledetection Environnementale -  Ecole Polytechnique Federale de Lausanne' 
    attrs['source'] = ''
    attrs['history'] = ''
    attrs['conventions'] = ''
    attrs['campaign_name'] = 'Common_2011'
    attrs['project_name'] = "",
    
    # - Instrument specs 
    attrs['sensor_name'] = "Parsivel"
    attrs["sensor_long_name"] = 'OTT Hydromet Parsivel'
    
    attrs["sensor_beam_width"] = 180     # TODO
    attrs["sensor_nominal_width"] = 180  # TODO
    attrs["measurement_interval"] = 30
    attrs["temporal_resolution"] = 30
    
    attrs["sensor_wavelegth"] = '650 nm'
    attrs["sensor_serial_number"] = ''
    attrs["firmware_IOP"] = ''
    attrs["firmware_DSP"] = ''
    
    # - Location info 
    attrs['station_id'] = [] # TODO
    attrs['station_name'] = [] # TODO
    attrs['station_number'] = [] # TODO
    attrs['location'] = [] # TODO
    attrs['country'] = "Switzerland"  
    attrs['continent'] = "Europe" 
 
    attrs['latitude'] = [0,0,0,0,0]  # TODO, Example [1,2,3,4,5]
    attrs['longitude'] = [0,0,0,0,0] # TODO, Example [1,2,3,4,5]
    attrs['altitude'] = [0,0,0,0,0]  # TODO, Example [1,2,3,4,5]
    
    attrs['latitude_unit'] = "DegreesNorth"   
    attrs['longitude_unit'] = "DegreesEast"   
    attrs['altitude_unit'] = "MetersAboveSeaLevel"   
    
    attrs['crs'] = "WGS84"   
    attrs['EPSG'] = 4326
    attrs['proj4_string'] = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

    # - Attribution 
    attrs['contributors'] = []
    attrs['authors'] = []
    attrs['reference'] = ''
    attrs['documentation'] = ''
    attrs['website'] = ''
    attrs['source_repository'] = ''
    attrs['doi'] = ''    
    attrs['contact'] = ''    
    attrs['contact_information'] = 'http://lte.epfl.ch' 
    
    # - DISDRODB attrs 
    attrs['source_data_format'] = 'raw_data'        
    attrs['obs_type'] = 'raw'   # preprocess/postprocessed
    attrs['level'] = 'L0'       # L0, L1, L2, ...    
    attrs['disdrodb_id'] = [20,21,22,40,41]   # TODO, Example   [20,21,22,40,41]        
 
    ##------------------------------------------------------------------------. 
    
    sensor_name = attrs['sensor_name']
    
    
    ###############################
    #### Perform L0 processing ####
    ###############################
    
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
    # Popolate device list
    device_list = {}
    i = 0
    for device in glob.glob(os.path.join(raw_dir,"data", "*")):
        try:
            device_list[os.path.basename(device)] = Sensor(
                        attrs['disdrodb_id'][i],
                        attrs['sensor_name'],
                        attrs["sensor_long_name"],
                        attrs["sensor_beam_width"],
                        attrs["sensor_nominal_width"],
                        attrs["measurement_interval"],
                        attrs["temporal_resolution"],
                        attrs["sensor_wavelegth"],
                        attrs["sensor_serial_number"],
                        attrs["firmware_IOP"],
                        attrs["firmware_DSP"],
                        attrs['station_id'],
                        attrs['station_name'],
                        attrs['station_number'],
                        attrs['location'],
                        attrs['country'],
                        attrs['continent'],
                        attrs['latitude'][i],
                        attrs['longitude'][i],
                        attrs['altitude'][i],
                        attrs['latitude_unit'] ,
                        attrs['longitude_unit'],
                        attrs['altitude_unit'],
                        attrs['crs'],
                        attrs['EPSG'],
                        attrs['proj4_string'])
        except IndexError:
            device_list[os.path.basename(device)] = Sensor(
                        os.path.basename(device),
                        attrs['sensor_name'],
                        attrs["sensor_long_name"],
                        attrs["sensor_beam_width"],
                        attrs["sensor_nominal_width"],
                        attrs["measurement_interval"],
                        attrs["temporal_resolution"],
                        attrs["sensor_wavelegth"],
                        attrs["sensor_serial_number"],
                        attrs["firmware_IOP"],
                        attrs["firmware_DSP"],
                        attrs['station_id'],
                        attrs['station_name'],
                        attrs['station_number'],
                        attrs['location'],
                        attrs['country'],
                        attrs['continent'],
                        attrs['latitude'],
                        attrs['longitude'],
                        attrs['altitude'],
                        attrs['latitude_unit'] ,
                        attrs['longitude_unit'],
                        attrs['altitude_unit'],
                        attrs['crs'],
                        attrs['EPSG'],
                        attrs['proj4_string'])
        i += 1
    
    ##------------------------------------------------------.   
    # Process all devices
    
    all_files = len(glob.glob(os.path.join(raw_dir, 'data', "**/*")))
    list_skipped_files = []
    
    msg = f'{all_files} files to process in {raw_dir}'
    if verbose:
        print(msg)
    logger.info(msg)
    
    for device in device_list:
        device_path = os.path.join(raw_dir,'data', device)
    # for device in range (0,1):
        
        # for device in 
        if L0_processing: 
            #----------------------------------------------------------------.
            t_i = time.time() 
            
            msg = f"L0 processing of device {device} started"
            if verbose:
                print(msg)
            logger.info(msg)
            # - Retrieve filepaths of raw data files 
            # file_list = sorted(glob.glob(os.path.join(raw_dir,"*")))
            # - With campaign path and all the stations files
            
            file_list = sorted(glob.glob(os.path.join(device_path,"**/*.dat*"), recursive = True))      
            # file_list = glob.glob(os.path.join(raw_dir,"nan_zip", "*"))
            
            if debug_on: 
                file_list = file_list[0:5]
            
            #----------------------------------------------------------------.
            # - Define raw data headers 
            
            raw_data_columns = ['time',
                                'id',
                                'sensor_heating_current',
                                'sensor_battery_voltage',
                                'unknow',
                                'unknow2',
                                'unknow3',
                                'unknow4',
                                'reflectivity_16bit',
                                'unknow5',
                                'A_voltage?', #Has flag -9.999
                                'unknow6',   #Has flag 9999
                                'sensor_temperature',  
                                'unknow7',
                                'A_voltage2?',
                                'unknow8',
                                'unknow9',
                                'Debug_data',
                                'FieldN',
                                'FieldV',
                                'RawData',
                                'All_0',
                                ]
            
            # time_col = ['time']
            
            check_valid_varname(raw_data_columns)
            
            ##------------------------------------------------------.      
            # Determine dtype based on standards 
            dtype_dict = get_dtype_standards_all_object()
            # dtype_dict = {column: dtype_dict[column] for column in raw_data_columns}
            
            dtype_temp = dtype_dict
            
            ##------------------------------------------------------.
            # Define reader options 
            reader_kwargs = {}
            # reader_kwargs['delimiter'] = ','
            reader_kwargs["on_bad_lines"] = 'skip'
            reader_kwargs["engine"] = 'python'
            # - Replace custom NA with standard flags 
            reader_kwargs['na_values'] = ['', 'error', 'NA']
            # Define time column
            # reader_kwargs['parse_dates'] = time_col
            reader_kwargs["blocksize"] = None
            # reader_kwargs["compression"] = 'gzip'
            
            ##------------------------------------------------------.
            # Loop over all files
            
            msg = f"{len(file_list)} files to process for {device}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            list_df = []
            for filename in file_list:
                try:
                    df = dd.read_csv(filename,
                                     dtype = dtype_temp,
                                     names = raw_data_columns,
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
                    
                    # Check column number
                    if len(df.columns) != len(raw_data_columns):
                        msg = f"{filename} has wrong columns number, and has been skipped"
                        logger.warning(msg)
                        if verbose: 
                            print(msg)
                        list_skipped_files.append(msg)
                        continue
                    
                    #-------------------------------------------------------------------------
                    ### Keep only clean data 
                    # Drop rows with more than 2 nan
                    df = df.dropna(thresh = (len(raw_data_columns) - 2), how = 'all')
                    
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
                        
                        
                    ##------------------------------------------------------.
                    # Check dtype
                    if dtype_check:
                        col_dtype_check(df, filename, verbose)

                    # - Append to the list of dataframe 
                    list_df.append(df)
                    
                    logger.debug(f'{filename} processed successfully')
                    
                except (Exception, ValueError) as e:
                  msg = f"{filename} has been skipped. The error is {e}"
                  logger.warning(f'{filename} has been skipped')
                  if verbose: 
                      print(msg)
                  list_skipped_files.append(msg)
            
            msg = f"{len(list_skipped_files)} files on {len(file_list)} for {device} has been skipped."
            if verbose:      
                print(msg)
            logger.info('---')
            logger.info(msg)
            logger.info('---')
                
            
            ##------------------------------------------------------.
            # Concatenate the dataframe 
            msg = f"Start concat dataframes for device {device}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            try:
                df = dd.concat(list_df, axis=0, ignore_index = True)
                # Drop duplicated values 
                df = df.drop_duplicates(subset="time")
                # Sort by increasing time 
                df = df.sort_values(by="time")
                
            except (AttributeError, TypeError) as e:
                msg = f"Can not create concat data files. Error: {e}"
                logger.exception(msg)
                raise ValueError(msg)
            
            msg = f"Finish concat dataframes for device {device}"
            if verbose:
                print(msg)
            logger.info(msg)
                
            ##------------------------------------------------------.                                   
            # Write to Parquet 
            msg = f"Starting conversion to parquet file for device {device}"
            if verbose:
                print(msg)
            logger.info(msg)
            # Path to device folder
            path = os.path.join(processed_path + '/' + device + '/L0/')
            # Write to Parquet 
            _write_to_parquet(df, path, campaign_name, force)
            
            msg = f"Finish conversion to parquet file for device {device}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##------------------------------------------------------.   
            # Check Parquet standards 
            check_L0_standards(df)
            
            ##------------------------------------------------------.   
            t_f = time.time() - t_i
            msg = "L0 processing of {} ended in {:.2f}s".format(device, t_f)
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##------------------------------------------------------.   
            # Delete temp variables
            del df
            del list_df
        
        #-------------------------------------------------------------------------.
        ###############################
        #### Perform L1 processing ####
        ###############################
        if L1_processing: 
            ##-----------------------------------------------------------
            t_i = time.time()  
            
            msg =f"L1 processing of device {device} started"
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##-----------------------------------------------------------
            # Check the L0 df is available 
            # Path device folder parquet
            df_fpath = os.path.join(processed_path + '/' + device + '/L0/' + campaign_name + '.parquet')
            if not L0_processing:
                if not os.path.exists(df_fpath):
                    msg = "Need to run L0 processing. The {df_fpath} file is not available."
                    logger.exception(msg)
                    raise ValueError(msg)
            
            msg = f'Found parquet file: {df_fpath}'
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##-----------------------------------------------------------
            # Read raw data from parquet file 
            msg = f'Start reading: {df_fpath}'
            if verbose:
                print(msg)
            logger.info(msg)
            
            df = dd.read_parquet(df_fpath)
            
            msg = f'Finish reading: {df_fpath}'
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##-----------------------------------------------------------
            # Subset row sample to debug 
            if not lazy and debug_on:
                df = df.iloc[0:100,:] # df.head(100) 
                df = df.iloc[0:,:]
                
                msg = 'Debug = True and Lazy = False, then only the first 100 rows are read'
                if verbose:
                    print(msg)
                logger.info(msg)
               
            ##-----------------------------------------------------------
            # Retrieve raw data matrix 
            msg = f"Retrieve raw data matrix for device {device}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            dict_data = {}
            n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
            
            if not lazy and debug_on:
                n_timesteps = df.shape[0]
            else:
                n_timesteps = df.shape[0].compute()
                
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
                    try:
                        da_arr = da_arr.reshape(n_timesteps, n_bins_dict['FieldN'], n_bins_dict['FieldV'])
                    except Exception as e:
                        msg = f'Error on retrive raw data matrix: {e}'
                        logger.error(msg)
                        print(msg)
                        # raise SystemExit
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
            
            msg = f"Finish retrieve raw data matrix for device {device}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##-----------------------------------------------------------
            # Define data variables for xarray Dataset 
            data_vars = {"FieldN": (["time", "diameter_bin_center"], dict_data['FieldN']),
                         "FieldV": (["time", "velocity_bin_center"], dict_data['FieldV']),
                         "RawData": (["time", "diameter_bin_center", "velocity_bin_center"], dict_data['RawData']),
                        }
            
            # Define coordinates for xarray Dataset
            coords = get_L1_coords(sensor_name=sensor_name)
            coords['time'] = df['time'].values
            
            coords['latitude'] = device_list[device].latitude
            coords['longitude'] = device_list[device].longitude
            coords['altitude'] = device_list[device].latitude
            coords['crs'] = device_list[device].crs
            coords['disdrodb_id'] = device_list[device].disdrodb_id
            
            attrs['latitude'] = device_list[device].latitude
            attrs['longitude'] = device_list[device].longitude
            attrs['altitude'] = device_list[device].latitude
            attrs['crs'] = device_list[device].crs
            attrs['disdrodb_id'] = device_list[device].disdrodb_id
            
            # coords['latitude'] = attrs['latitude']
            # coords['longitude'] = attrs['longitude']
            # coords['altitude'] = attrs['altitude']
            # coords['crs'] = attrs['crs']
    
            ##-----------------------------------------------------------
            # Create xarray Dataset
            try:
                ds = xr.Dataset(data_vars = data_vars, 
                                coords = coords, 
                                attrs = attrs,
                                )
            except Exception as e:
                msg = f'Error on creation xarray dataset: {e}'
                logger.error(msg)
                print(msg)
                # raise SystemExit
            
            ##-----------------------------------------------------------
            # Check L1 standards 
            check_L1_standards(ds)
            
            ##-----------------------------------------------------------    
            # Write to Zarr as intermediate storage 
            if keep_zarr:
                tmp_zarr_fpath = os.path.join(processed_path + '/' + device + '/L1/' + campaign_name + '.zarr')
                ds = rechunk_L1_dataset(ds, sensor_name=sensor_name)
                zarr_encoding_dict = get_L1_zarr_encodings_standards(sensor_name=sensor_name)
                ds.to_zarr(tmp_zarr_fpath, encoding=zarr_encoding_dict, mode = "w")
            
            ##-----------------------------------------------------------  
            # Write L1 dataset to netCDF
            # Path for save into device folder
            path = os.path.join(processed_path + '/' + device)
            L1_nc_fpath = path + '/L1/' + campaign_name + '.nc'
            ds = rechunk_L1_dataset(ds, sensor_name=sensor_name) # very important for fast writing !!!
            nc_encoding_dict = get_L1_nc_encodings_standards(sensor_name=sensor_name)
            
            if debug_on:
                ds.to_netcdf(L1_nc_fpath, engine="netcdf4")
            else:
                ds.to_netcdf(L1_nc_fpath, engine="netcdf4", encoding=nc_encoding_dict)
            
            ##-----------------------------------------------------------
            t_f = time.time() - t_i
            msg = "L1 processing of device {} ended in {:.2f}s".format(device, t_f)
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
    # main() # when using click 
    main(raw_dir, processed_path, L0_processing, L1_processing, force, verbose, debug_on, lazy, keep_zarr, dtype_check)
