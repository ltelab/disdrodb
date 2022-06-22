#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:45:24 2022

@author: ghiggi
"""
##----------------------------------------------------------------------------.
### Sensor status  
# 0: "Everything ok"
# 1: "Laser protective glass is dirty, but measurement is still possible"
# 2: "Laser protective glass is dirty, partially covered. No measurement possible"
# 3: "Laser damaged. No measurement possible"
# --> TODO: Add to explanations 

### Error code  
# --> TODO: Add to explanations 

##---------------------------------------------------------.
### Refactoring and Code Skeleton 
# diameter_bins.yml --> bins_diameter.yml 
# velocity_bins.yml --> bins_velocity.yml 
# variable_explanations.yml --> variable_description.yml 

# For ThiesLPM, add info in variables_units.yml --> Look in dev/OLD_standards.py
# For ThiesLPM, add info and the file variables_explanations.yml --> Look in dev/OLD_standards.py --> "description"
# For OTT_Parsivel configs, check that valid_min, valid_max in dev/OLD_standards.py correspond to our configs... LET'S DISCUSS DIFFERENCES

## L0_dtype.yaml : why "object"? works also with "str" ?

##---------------------------------------------------------.
# Renaming
# - get_L1_netcdf_encoding_dict --> get_L0_encoding_dict
# - get_L0_dtype_standards --> get_L0_dtype_raw

##---------------------------------------------------------.
### Standards
# Exploit nan_flags in L0_data_format.yml 
# Check data_range for each variable
#  --> Outside range set to NaN and log the occurence 
# Check nan_flags for each variable
#  --> Replace with NaN
#  --> Log the occurence

# Check actual data_range 
# --> Open processed netCDF and look at histogram (or unique counts) 

##---------------------------------------------------------.
### Bins 
 
# bnds dimension in netCDF? 
# NASA bins to be checked  
# see dev/NASA_bins.py

##----------------------------------------------------------------------------.
### Synop weather codes dictionaries
# - Add synop weather code tables (as yml file) in disdrodb/weather_code folder
# - Take from manual of OTT and LPM 
# - Check against: http://www.czo.psu.edu/downloads/Metadataworksheets/LPM_SYNOP_METAR_key.pdf

 