#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:45:24 2022

@author: ghiggi
"""
##---------------------------------------------------------.
# Remove lat/lon dai dataframe
# - A meno di dataset speciali dove il sensore si muove (i.e. nave, funivia,...)

# SAMYLONOV: parser da ricontrollare
# - date_time_measuring_start? really?
# - snowfall intensity? exists in Parsivel1 only ...

# COMMOM_2011:  df = df[df.Debug_data.str.contains("Frame") == False]
# --> Never do == True/False ... use np.invert to change from True to False (and viceversa)
# --> You can index with array of booleans

### Rename /ltenas3/0_Data/ParsivelDB to DISDRODB

### Check/adjust the script to run all EPFL Parsers

### Ensure that for each parser_<campaign_name>.py there is a single data directory !!!

##---------------------------------------------------------.
### Change variable names to new standards (i.e. FieldN, FieldV and RawData)
# --> All variable name to be lowercase!
# weather_code_SYNOP --> snyop, metar, all to lowercase
# OTT_Parsivel2: snowfall_intensity → snowfall_rate
# OTT_Parsivel2 and OTT_Parsivel: rain_* → rainfall_* , n_particles(_all) → number_particles(_all)

##---------------------------------------------------------.
## L0_dtype.yaml : why "object"? works also with "str" ?

## Add synop weather code tables (as yml file) in disdrodb/weather_code folder

## Double check values in L0_data_formats.yaml
## Check data_range and nan_flags for each variable
#  --> You could check for the min,max of each variable already saved to netCDF ... (ds.min() , ds.max())

# Add units and explanations to yaml files

##---------------------------------------------------------.
## Refactor data_encodings.py -->

## Add DIVEN and ARM dictionary + Conversion to netCDF

##---------------------------------------------------------.
## Add yaml for LPMs

### Parser for DELFT LPM csv and the 2 other links in slack

##---------------------------------------------------------.
### Others
# https://docs.google.com/document/d/1CiZ-Xo3NRy5FxTHUxh-uZ47A9mRAU6qETiyXHEyxJ60/edit
