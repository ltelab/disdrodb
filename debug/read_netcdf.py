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


path = r'/SharedVM/Campagne/ltnas3/Processed/Payerne_2014/20/l1'
campagna = 'Payerne_2014'
file = campagna + '.nc'
file_path = os.path.join(path, file)

ds = xr.open_dataset(file_path)
df = ds.to_dataframe()
