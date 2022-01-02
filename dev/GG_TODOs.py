#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:42:14 2022

@author: ghiggi
"""  
#### L0 checks 
# in dev directory ... 

####--------------------------------------------------------------------------.
#### L1_proc.py
create_L1_dataset_from_L0 
# - add auxiliary data 
# - replace NA flags 
write_L1_to_zarr    # rechunk before writing 
# - add zarr encoding 
write_L1_to_netcdf
# - add nc encoding 
create_L1_summary_statistics
# - regularize timeseries 
# - number of dry/rainy minutes 
# - timebar plot with 0,>1, NA, no data rain rate (ARM STYLE) 
# - timebar data quality 

####--------------------------------------------------------------------------.
# reformat_ARM_LPM 
# reformat_DIVEN_LPM 

# metadata.py 
check_metadata_compliance # check dtype also, raise errors !

# L0_proc.py
# - Filter bad data based on sensor_status/error_code
# - check df_sanitizer_fun has only lazy and df arguments ! 
# - Implement check_L0_standards in write_df_to_parquet

# - Add DISDRODB attrs 
attrs['source_data_format'] = 'raw_data'        
attrs['obs_type'] = 'raw'   # preprocess/postprocessed
attrs['level'] = 'L0'       # L0, L1, L2, ...    
attrs['disdrodb_id'] = ''   # TODO     
 
####--------------------------------------------------------------------------.
### Others  
# - Template work for Ticino, Payerne, 1 ARM Parsivel, 1 UK Diven, 1 Hymex 
# - Copy metadata yaml to github for syncs and review by external 