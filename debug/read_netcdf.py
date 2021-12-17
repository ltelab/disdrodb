#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:49:23 2021

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import os
import xarray as xr
import netCDF4

path = r'/SharedVM/Campagne/ltnas3/Processed/Parsivel_2007/L1'
campagna = 'Parsivel_2007'
file = campagna + '.nc'
file_path = os.path.join(path, file)

f3 = netCDF4.Dataset(file_path)


path = r'/SharedVM/Campagne/ltnas3/Processed/Payerne_2014/10/L1'
campagna = 'Payerne_2014'
file = campagna + '.nc'
file_path = os.path.join(path, file)

f = netCDF4.Dataset(file_path)

path = r'/SharedVM/Campagne/ltnas3/Processed/Payerne_2014/20/L1'
campagna = 'Payerne_2014'
file = campagna + '.nc'
file_path = os.path.join(path, file)

f2 = netCDF4.Dataset(file_path)