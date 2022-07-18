#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:22:02 2022

@author: ghiggi
"""
# Metadata attributes 
# - source_data_type: 'raw', 'preprocessed'    # or data_source_type
# - source_data_format: 'raw', 'nc',           # or data_source_format

# - deployment_status: 'ended', 'ongoing'
# - deployment mode: 'land', 'ship', 'truck', 'cable'   
# - platform_type: ' stationary', 'mobile'
# - platform_protection: 'N/A', 'shielded', 'unshielded'
# - comments
# - acknowledgments 
# - license

# - sampling_interval  [temporal_resolution vs. sample_interval vs. measurement_interval]   [in seconds]

# - effective_measurement_area  # 0.54 m^2
# - "sensor_beam_width" vs  "sensor_nominal_width"  
# - "sensor_beam_length"

# DISDRODB Global 
# - processing_date
# - disdrodb_software_version
# - disdrodb_product_version
# - disdrodb_product_level
# - disdrodb_id

# VARIABLES 
# - long_name 
# - description
# - units
# - valid_min 
# - valid_max 
# - flag_values  or valid_flag_values (status, error code ...)
# - flag_name/meaning
# - variable_type: coordinate, count, category, quality_flag, quantity, flux  

# COORDS LAT/LON/TIME
# - CRS, SPATIAL_REF GRID_MAPPING  

# COORDS DIAMETER/VELOCITY
# --> diameter/velocity dimension?  or diameter_bin_center/velocity_bin_center?
 
# --> BOUNDS/NV dimension? -
#     --> diameter_bnds or 2 variables: diameter_bin_lower, diameter_bin_upper
# --> diameter_bin_spread or diameter_bin_width

# COORDS: TIME BNDS ? 


#-----------------------------------------------------------------------------.
# TODO 
# - station_id 
# - station_number --> to be replaced by station_id 
# - station_name --> if missing, use station_id 
# - raise error if missing campaign_name 
# - raise error if missing station_id & station_name  
# - Add station_name to all yaml 