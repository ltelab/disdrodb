#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:27:30 2022

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import os
import xarray as xr
import netCDF4

from disdrodb.data_encodings import get_DIVEN_to_l0_dtype_standards
from disdrodb.standards import get_var_explanations_ARM

dict_DIVEN = get_DIVEN_to_l0_dtype_standards()

file_path = '/SharedVM/Campagne/DIVEN/Raw/data/cairngorm/2017/02/ncas-disdrometer-11_cairngorm_20170210_precipitation_v1.0.nc'

ds = xr.open_dataset(file_path)

print(list(ds.keys()))

ds = ds.rename(dict_DIVEN)

print(list(ds.keys()))

ds.close()




