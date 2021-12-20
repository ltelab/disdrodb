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

campagna = 'COMMON_2011'
path = f'/SharedVM/Campagne/ltnas3/Processed/{campagna}/20/L1'
file = campagna + '.nc'
file_path = os.path.join(path, file)

f = netCDF4.Dataset(file_path)

print(f)

f.close()

# campagna = 'Parsivel_2007'
# path = f'/SharedVM/Campagne/ltnas3/Processed/{campagna}/L1'
# file = campagna + '.nc'
# file_path = os.path.join(path, file)

# f2 = netCDF4.Dataset(file_path)

# print(f2)

# f2.close()