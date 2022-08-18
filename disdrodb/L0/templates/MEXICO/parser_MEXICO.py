#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:08:10 2022

@author: ghiggi
"""
import os
import glob
import xarray as xr

dir_path = "/ltenas3/0_Data/DISDRODB/TODO_Raw/MEXICO/OH_IIUNAM/data"
fpaths = glob.glob(os.path.join(dir_path, "*.nc"))
fpath = fpaths[0]
 
ds = xr.open_dataset(fpath)

 