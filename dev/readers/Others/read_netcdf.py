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
from pprint import pprint


file_path = '/home/kimbo/data/Campagne/Processed/MELBURNE/MELBOURNE_2007_THIES/L1/MELBOURNE_2007_THIES_s1.nc'

ds = xr.open_dataset(file_path)

print(ds)





























