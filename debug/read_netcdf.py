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

campagna = 'DAVOS_2009'
path = f"/SharedVM/Campagne/ltnas3/Processed/{campagna}/50_incoming/L1"
file = campagna + '.nc'
file_path = os.path.join(path, file)

ds = xr.open_dataset(file_path)

# ds['time'] = ds['time'].astype('M8')

# ds['FieldN'].plot(x = 'time', y = 'diameter_bin_center')

# ds.plot(x = 'time', y = 'diameter_bin_center')


print(ds)

ds.close()