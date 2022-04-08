#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:03:26 2022

@author: kimbo
"""

import os
import xarray as xr

import logging
import time
import click

# Encodings
from disdrodb.data_encodings import get_ARM_LPM_dict
from disdrodb.data_encodings import get_ARM_LPM_dims_dict


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
from disdrodb.io import get_L1_netcdf_fpath

# L1_processing
from disdrodb.L1_proc import write_L1_to_netcdf


############## 


### Function ###

    
def compare_standard_keys(dict_campaing, ds_keys, verbose):
    '''Compare a list (NetCDF keys) and a dictionary (standard from a campaing keys) and rename it, if a key is missin into the dictionary, take the missing key and add the suffix _OldName.'''
    dict_standard = {}
    # Initial list skipped keys
    list_skipped_keys = []

    # Shorter version
    # dict_standard = {k: dict_campaing[k] for k in dict_campaing if k in ds_keys}
    
    # Loop keys
    for ds_v in ds_keys:
        try:
            # Check if key into dict_campaing
            dict_standard[ds_v] = dict_campaing[ds_v]
        except KeyError:
            # If not present, give non standard name and add to list_skipped_keys
            dict_standard[ds_v] = ds_v + '_________TO_CHECK_VALUE_INTO_DATA_ENCONDINGS'
            list_skipped_keys.append(ds_v)
            pass
                
    # Log
    if list_skipped_keys:
        msg = f"Cannot convert keys values: {len(list_skipped_keys)} on {len(ds_keys)} \n Missing keys: {list_skipped_keys}"
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
    
    # Compare NetCDF and dictionary keys
    dict_checked = compare_standard_keys(dict_campaign, ds_keys, verbose)
    
    return dict_checked

def reformat_ARM_files(file_list, processed_dir, attrs, verbose):
    '''
    file_list:      List of NetCDF's path with the same ID
    processed_dir:  Save location for the renamed NetCDF
    attrs:          Info about campaing
    verbose:        Flag for more information on terminal output
    '''
    
    from disdrodb.L1_proc import get_L1_coords
    
    dict_ARM = get_ARM_LPM_dict()
    # Custom dictonary for the campaign defined in standards
    dict_campaign = create_standard_dict(file_list[0], dict_ARM, verbose)
    
    # Open netCDFs
    file_list = sorted(file_list)
    ds = xr.open_mfdataset(file_list)
    
    # Get coords
    coords = get_L1_coords(attrs['sensor_name'])
    
    # Assign coords and attrs
    coords["crs"] = attrs["crs"]
    coords["altitude"] = attrs["altitude"]
    
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
        # Assign coords
        ds = ds.assign_coords(coords)
        ds.attrs = attrs
    
    except Exception as e:
        msg = f"Error in rename variable. The error is: \n {e}"
        raise RuntimeError(msg)
        # To implement when move fuction into another file, temporary solution for now
        # logger.error(msg)
        # raise RuntimeError(msg)
        
    # Delete some keys for temporary solution for temp keys names (then to add into yml encodings)
    data_vars_to_drop = ['latitude',
                        'longitude',
                        'altitude',
                        'base_time_calculated_ToConfirmIntoData_encodings',
                        'time_offset_calculated_ToConfirmIntoData_encodings',
                        'time_bounds_calculated_ToConfirmIntoData_encodings',
                        'particle_diameter_bounds_calculated_ToConfirmIntoData_encodings',
                        'particle_fall_velocity_bounds_ToConfirmIntoData_encodings',
                        'particle_fall_velocity_bounds_calculated_ToConfirmIntoData_encodings',
                        'air_temperature_calculated_ToConfirmIntoData_encodings',
                        'qc_time_calculated_ToConfirmIntoData_encodings',
                        'equivalent_radar_reflectivity_ott_calculated_ToConfirmIntoData_encodings',
                        'class_size_width_calculated_ToConfirmIntoData_encodings',
                        'fall_velocity_calculated_ToConfirmIntoData_encodings',
                        'liquid_water_content_calculated_ToConfirmIntoData_encodings',
                        'intercept_parameter_calculated_ToConfirmIntoData_encodings',
                        'slope_parameter_calculated_ToConfirmIntoData_encodings',
                        'median_volume_diameter_calculated_ToConfirmIntoData_encodings',
                        'liquid_water_distribution_mean_calculated_ToConfirmIntoData_encodings',
                        'diameter_min_calculated_ToConfirmIntoData_encodings',
                        'diameter_max_calculated_ToConfirmIntoData_encodings',
                        'diameter_min_ToConfirmIntoData_encodings',
                        'diameter_max_ToConfirmIntoData_encodings',
                        'moment1_calculated_ToConfirmIntoData_encodings',
                        'moment2_calculated_ToConfirmIntoData_encodings',
                        'moment3_calculated_ToConfirmIntoData_encodings',
                        'moment4_calculated_ToConfirmIntoData_encodings',
                        'moment5_calculated_ToConfirmIntoData_encodings',
                        'moment6_calculated_ToConfirmIntoData_encodings',
                        'moment1_ToConfirmIntoData_encodings',
                        'moment2_ToConfirmIntoData_encodings',
                        'moment3_ToConfirmIntoData_encodings',
                        'moment4_ToConfirmIntoData_encodings',
                        'moment5_ToConfirmIntoData_encodings',
                        'moment6_ToConfirmIntoData_encodings',
                        'qc_precip_rate_calculated_ToConfirmIntoData_encodings',
                        'qc_weather_code_calculated_ToConfirmIntoData_encodings',
                        'qc_equivalent_radar_reflectivity_ott_calculated_ToConfirmIntoData_encodings',
                        'qc_mor_visibility_calculated_ToConfirmIntoData_encodings',
                        'qc_snow_depth_intensity_calculated_ToConfirmIntoData_encodings',
                        'qc_laserband_amplitude_calculated_ToConfirmIntoData_encodings',
                        'qc_heating_current_calculated_ToConfirmIntoData_encodings',
                        'qc_sensor_voltage_calculated_ToConfirmIntoData_encodings',
                        'qc_number_detected_particles_calculated_ToConfirmIntoData_encodings',
                        ]
    
    a = list(ds.keys())
    
    c = set(a).intersection(data_vars_to_drop)
    
    try:
        ds = ds.drop(c)
    except Exception as e:
        msg = f"Error in rename variable. The error is: \n {e}"
        raise RuntimeError(msg)
    
    # Close NetCDF
    ds.close()
        
    return ds


### Script ###

# -------------------------------------------------------------------------.
# CLIck Command Line Interface decorator
@click.command()  # options_metavar='<options>'
@click.argument('raw_dir', type=click.Path(exists=True), metavar='<raw_dir>')
@click.argument('processed_dir', metavar='<processed_dir>')
@click.option('-l0', '--l0_processing', type=bool, show_default=False, default=True, help="Perform L0 processing")
@click.option('-l1', '--l1_processing', type=bool, show_default=False, default=True, help="Perform L1 processing")
@click.option('-nc', '--write_netcdf', type=bool, show_default=True, default=True, help="Write L1 netCDF4")
@click.option('-f', '--force', type=bool, show_default=True, default=False, help="Force overwriting")
@click.option('-v', '--verbose', type=bool, show_default=True, default=False, help="Verbose")
@click.option('-d', '--debugging_mode', type=bool, show_default=True, default=False, help="Switch to debugging mode")
@click.option('-l', '--lazy', type=bool, show_default=True, default=True, help="Use dask if lazy=True")
def main(raw_dir,
         processed_dir,
         l0_processing=False,
         l1_processing=False,
         write_netcdf=False,
         force=True,
         verbose=True,
         debugging_mode=False,
         lazy=True
         ):


    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    raw_data_glob_pattern= "*.cdf*"
    extension_file = ["*.cdf*", "*.nc*"]
    
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

            
        # Start rename processing
        t_i = time.time()
        msg = " - Rename NetCDF of station_id {} has started.".format(station_id)
        if verbose:
            print(msg)
        logger.info(msg)

        # -----------------------------------------------------------------.
        #### - List files to process
        if extension_file:
            glob_pattern = os.path.join("data", station_id, raw_data_glob_pattern)
        else:
            glob_pattern = os.path.join("data", station_id)
        file_list = get_file_list(
            raw_dir=raw_dir,
            glob_pattern=glob_pattern,
            verbose=verbose,
            debugging_mode=debugging_mode,
            extension_file=extension_file
        )
        
        # Rename variable netCDF
        ds = reformat_ARM_files(file_list, processed_dir, attrs, verbose)
        
        fpath = get_L1_netcdf_fpath(processed_dir, station_id)
        write_L1_to_netcdf(ds, fpath=fpath, sensor_name=sensor_name)
        # Temp for debug purpose
        # ds.to_netcdf(fpath, engine="netcdf4")
        
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
        
    
    msg = "### Script finish ###"
    print("\n  " + msg + "\n")
    logger.info(msg)
    
    close_logger(logger)

#################################

if __name__ == "__main__":
    main()
    # main(raw_dir = "/SharedVM/Campagne/ARM/Raw/SOUTHWEST_PACFIC_OCEAN",
    #     processed_dir = "/SharedVM/Campagne/ARM/Processed/SOUTHWEST_PACFIC_OCEAN",
    #     l0_processing=False,
    #     l1_processing=False,
    #     write_netcdf=False,
    #     force=True,
    #     verbose=True,
    #     debugging_mode=True,
    #     lazy=True
    #     )



