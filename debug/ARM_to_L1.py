#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:26:01 2022

@author: kimbo
"""

import pandas as pd
import dask.dataframe as dd
import os
import xarray as xr
import netCDF4

from disdrodb.data_encodings import get_ARM_to_l0_dtype_standards
from disdrodb.standards import get_var_explanations_ARM

dict_ARM = get_ARM_to_l0_dtype_standards()

campagna = 'anxldM1.b1.20191201.000000'
path = "/SharedVM/Campagne/ARM/anxldM1"
file = campagna + '.cdf'
file_path = os.path.join(path, file)

ds = xr.open_dataset(file_path)

print(list(ds.keys()))

ds = ds.rename(dict_ARM)

print(list(ds.keys()))

ds.close()




