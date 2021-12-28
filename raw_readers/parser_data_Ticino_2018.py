#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:09:19 2021

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

### TODO 4 Kimbo tomorrow

# To think eventually 
# --> Intermediate storage in Zarr: 74.32s
# --> Direct writing to netCDF: 61s

## Infer Arrow schema from pandas
# import pyarrow as pa
# schema = pa.Schema.from_pandas(df)

# Error file busy on zarr, don't overwrite

# Attributes for every devices

# Make template work for Ticino, Payerne, 1 ARM Parsivel, 1 UK Diven, 1 Hymex


#-------------------------------------------------------------------------.
# Click implementation

@click.command(options_metavar='<options>')

@click.argument('raw_dir', type=click.Path(exists=True), metavar ='<raw_dir>')

@click.argument('processed_path', metavar ='<processed_path>') #TODO

@click.option('--l0_processing',    '--l0',     is_flag=True, show_default=True, default = False,   help = 'Process the campaign in l0_processing')
@click.option('--l1_processing',    '--l1',     is_flag=True, show_default=True, default = False,   help = "Process the campaign in l1_processing")
@click.option('--force',            '--f',      is_flag=True, show_default=True, default = False,   help = "Force ...")
@click.option('--verbose',          '--v',      is_flag=True, show_default=True, default = False,   help = "Verbose ...")
@click.option('--debug_on',         '--d',      is_flag=True, show_default=True, default = False,   help = "Debug ...")
@click.option('--lazy',             '--l',      is_flag=True, show_default=True, default = True,    help = "Lazy ...")
@click.option('--keep_zarr',        '--kz',     is_flag=True, show_default=True, default = False,   help = "Keep zarr ...")
@click.option('--dtype_check',        '--dc',     is_flag=True, show_default=True, default = False,   help = "Check if the data are in the standars (max lenght, data range) ...")

# raw_dir = "/SharedVM/Campagne/ltnas3/Raw/Ticino_2018/epfl_parsivel/raw"
# processed_path = '/SharedVM/Campagne/ltnas3/Processed/Ticino_2018'
# l0_processing = True
# l1_processing = False
# force = True
# verbose = True
# debug_on = True
# lazy = True
# keep_zarr = False
# dtype_check = False


#-------------------------------------------------------------------------.


def main(raw_dir, processed_path, l0_processing, l1_processing, force, verbose, debug_on, lazy, keep_zarr, dtype_check):
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
    attrs['title'] = 'DSD observations along a mountain transect near Locarno, Ticino, Switerland'
    attrs['description'] = '' 
    attrs['institution'] = 'Laboratoire de Teledetection Environnementale -  Ecole Polytechnique Federale de Lausanne' 
    attrs['source'] = ''
    attrs['history'] = ''
    attrs['conventions'] = ''
    attrs['campaign_name'] = 'Locarno2018'
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
    attrs['station_id'] = "" # TODO
    attrs['station_name'] = "" # TODO
    attrs['station_number'] = 0 # TODO
    attrs['location'] = "" # TODO
    attrs['country'] = "Switzerland"  
    attrs['continent'] = "Europe" 
 
    attrs['latitude'] = 0   # TODO
    attrs['longitude'] = 0  # TODO
    attrs['altitude'] = 0  # TODO
    
    attrs['latitude_unit'] = "DegreesNorth"   
    attrs['longitude_unit'] = "DegreesEast"   
    attrs['altitude_unit'] = "MetersAboveSeaLevel"   
    
    attrs['crs'] = "WGS84"   
    attrs['EPSG'] = 4326
    attrs['proj4_string'] = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

    # - Attribution 
    attrs['contributors'] = ''
    attrs['authors'] = ''
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
    attrs['disdrodb_id'] = ''   # TODO         
 
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
    logger = log(processed_path, 'template_dev')
    
    msg = '### Script start ###'
    print(msg)
    logger.info(msg)
    
    ##------------------------------------------------------.   
    # Process all devices
    
    all_files = len(glob.glob(os.path.join(raw_dir, "*/*/*.dat*")))
    list_skipped_files = []
    count_file = 0
    
    msg = f'{all_files} files to process in {raw_dir}'
    if verbose:
        print(msg)
    logger.info(msg)
    
    file_list = sorted(glob.glob(os.path.join(raw_dir, "*/*/*.dat*"))) 
    # file_list = glob.glob(os.path.join(raw_dir,"nan_zip", "*"))
    
    if debug_on: 
        file_list = file_list[0:5]
        
    # for device in 
    if l0_processing: 
        #----------------------------------------------------------------.
        t_i = time.time() 
        msg = "L0 processing of {} started".format(attrs['disdrodb_id'])
        if verbose:
            print(msg)
        logger.info(msg)
        # - Retrieve filepaths of raw data files 
        # file_list = sorted(glob.glob(os.path.join(raw_dir,"*")))
        # - With campaign path and all the stations files
        
        
        
        #----------------------------------------------------------------.
        # - Define raw data headers 
        # In all datafile, the datalogger voltage hasn't the delimeter, so need it to split (x)
        raw_data_columns = ['id',
                            'latitude',
                            'longitude',
                            'time',
                            'datalogger_temperature',
                            'temp', #datalogger_voltage rain_rate_32bit
                            'rain_accumulated_32bit',
                            'weather_code_SYNOP_4680',
                            'weather_code_SYNOP_4677',
                            'reflectivity_16bit',
                            'mor_visibility',
                            'laser_amplitude',  
                            'n_particles',
                            'sensor_temperature',
                            'sensor_heating_current',
                            'sensor_battery_voltage',
                            'sensor_status',
                            'rain_amount_absolute_32bit',
                            'error_code',
                            'FieldN',
                            'FieldV',
                            'RawData',
                            'datalogger_error'
                            ]
        
        time_col = ['time']
        
        check_valid_varname(raw_data_columns)
        
        ##------------------------------------------------------.
        # Define reader options 
        reader_kwargs = {}
        # reader_kwargs['compression'] = 'gzip'
        reader_kwargs['delimiter'] = ';'
        reader_kwargs["on_bad_lines"] = 'skip'
        reader_kwargs["engine"] = 'python'
        # - Replace custom NA with standard flags 
        reader_kwargs['na_values'] = 'na'
        # Define time column
        reader_kwargs['parse_dates'] = time_col
        if lazy:
            reader_kwargs["blocksize"] = None
        
        ##------------------------------------------------------.
        # Loop over all files
        
        # msg = f"{len(file_list)} files to process for {attrs['disdrodb_id']}"
        # if verbose:
        #     print(msg)
        # logger.info(msg)
        
        
        
        # dtype_dict = {column: dtype_dict[column] for column in raw_data_columns}
        
        list_df = []
        for filename in file_list:
                try:
                    df = dd.read_csv(filename,
                                     names = raw_data_columns,
                                     **reader_kwargs)
                    
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
                    
                    # Split column
                    df2 = df['temp'].str.split(',', n=1, expand=True)
                    
                    df2.columns = ['datalogger_voltage','rain_rate_32bit']

                    df = df.drop(columns=['temp'])

                    df = dd.concat([df,df2], axis = 1, ignore_unknown_divisions=True)
                    
                    #-------------------------------------------------------------------------
                    ### Keep only clean data 
                    # Drop latitude and longitute (always the same)
                    df = df.drop(columns=['latitude', 'longitude'])
                    
                    # Drop rows with half nan values
                    df = df.dropna(thresh = (len(df.columns) - 10), how = 'all')
                    
                    # Remove rows with bad data 
                    df = df[df.sensor_status == 0]
                    # Remove rows with error_code not 000 
                    df = df[df.error_code == 0]
                    
                    ##------------------------------------------------------.
                    # Cast dataframe to dtypes
                    # Determine dtype based on standards 
                    dtype_dict = get_L0_dtype_standards()
                    dtype_dict = {column: dtype_dict[column] for column in df.columns}

                    for k, v in dtype_dict.items():
                        df[k] = df[k].astype(v)
                        
                        
                    ##------------------------------------------------------.
                    # Check dtype
                    if dtype_check:
                        col_dtype_check(df, filename, verbose)
                
                    list_df.append(df)
                    
                    count_file += 1
                    
                    logger.debug(f'{count_file} on {all_files} processed successfully: File name: {filename}')
                    
                except (Exception, ValueError) as e:
                  msg = f"{filename} has been skipped. The error is {e}"
                  logger.warning(msg)
                  if verbose: 
                      print(msg)
                  list_skipped_files.append(msg)


        msg = f"{len(list_skipped_files)} files on {len(file_list)} for {attrs['disdrodb_id']} has been skipped."
        if verbose:      
            print(msg)
        logger.info('---')
        logger.info(msg)
        logger.info('---')
           
   
        ##------------------------------------------------------.
        # Concatenate the dataframe
        msg = f"Start concat dataframes for {attrs['disdrodb_id']}"
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
            
        msg = f"Finish concat dataframes for {attrs['disdrodb_id']}"
        if verbose:
            print(msg)
        logger.info(msg)     
       
            
        ##------------------------------------------------------.                                   
        # Write to Parquet 
        msg = f"Starting conversion to parquet file for {attrs['disdrodb_id']}"
        if verbose:
            print(msg)
        logger.info(msg)
        # Path to device folder
        path = os.path.join(processed_path)
        # Write to Parquet 
        _write_to_parquet(df, path, campaign_name, force)
        
        msg = f"Finish conversion to parquet file for {attrs['disdrodb_id']}"
        if verbose:
            print(msg)
        logger.info(msg)
        
        ##------------------------------------------------------.   
        # Check Parquet standards 
        check_L0_standards(df)
        
        ##------------------------------------------------------.   
        t_f = time.time() - t_i
        msg = "L0 processing of {} ended in {:.2f}s".format(attrs['disdrodb_id'], t_f)
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
    if l1_processing: 
            ##-----------------------------------------------------------
            t_i = time.time()
            msg = "L1 processing of {} started".format(attrs['disdrodb_id'])
            if verbose:
                print(msg)
            logger.info(msg)
            
            ##-----------------------------------------------------------
            # Check the L0 df is available 
            # Path device folder parquet
            df_fpath = os.path.join(processed_path + '/' + campaign_name + '.parquet')
            if not l0_processing:
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
            msg = f"Retrieve raw data matrix for {attrs['disdrodb_id']}"
            if verbose:
                print(msg)
            logger.info(msg)
            
            dict_data = {}
            n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
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
            
            msg = f"Finish retrieve raw data matrix for {attrs['disdrodb_id']}"
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
            coords['latitude'] = attrs['latitude']
            coords['longitude'] = attrs['longitude']
            coords['altitude'] = attrs['altitude']
            coords['crs'] = attrs['crs']
    
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
                tmp_zarr_fpath = os.path.join(processed_path + '/' + campaign_name + '.zarr')
                ds = rechunk_L1_dataset(ds, sensor_name=sensor_name)
                zarr_encoding_dict = get_L1_zarr_encodings_standards(sensor_name=sensor_name)
                ds.to_zarr(tmp_zarr_fpath, encoding=zarr_encoding_dict, mode = "w")
            
            ##-----------------------------------------------------------  
            # Write L1 dataset to netCDF
            # Path for save into device folder
            path = os.path.join(processed_path)
            L1_nc_fpath = path + '/' + campaign_name + '.nc'
            ds = rechunk_L1_dataset(ds, sensor_name=sensor_name) # very important for fast writing !!!
            nc_encoding_dict = get_L1_nc_encodings_standards(sensor_name=sensor_name)
            
            if debug_on:
                ds.to_netcdf(L1_nc_fpath, engine="netcdf4")
            else:
                ds.to_netcdf(L1_nc_fpath, engine="netcdf4", encoding=nc_encoding_dict)
            
            
            ##-----------------------------------------------------------
            try:
                msg = "L1 processing of {} ended in {:.2f}s".format(attrs['disdrodb_id'], t_f)
                if verbose:
                    t_f = time.time() - t_i
                    print(msg)
                logger.info(msg)
            except Exception:
                pass
        
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
    # main(raw_dir, processed_path, l0_processing, l1_processing, force, verbose, debug_on, lazy, keep_zarr, dtype_check)
    
 
