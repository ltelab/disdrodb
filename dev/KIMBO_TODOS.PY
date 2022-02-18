#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:45:24 2022

@author: ghiggi
"""

# Remove lat/lon dai dataframe
# - A meno di dataset speciali dove il sensore si muove (i.e. nave, funivia,...)

# SAMYLONOV: parser da ricontrollare

# COMMOM_2011:  df = df[df.Debug_data.str.contains("Frame") == False]
# --> Never do == True/False ... use np.invert to change from True to False (and viceversa)
# --> You can index with array of booleans

# PLATO_2019: no fieldn,v, raw data


## same variable name for FieldN, FieldV and RawData across all sensors


### In L1, add attribute indicating if dataset contains raw_spectrum
# - Nd and Vd are derived
