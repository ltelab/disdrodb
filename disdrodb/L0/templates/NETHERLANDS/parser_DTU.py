#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:05:47 2022

@author: ghiggi
"""
import os
import glob
import xarray as xr

dir_path = "/ltenas3/0_Data/DISDRODB/TODO_Raw/DENMARK/EROSION/data"
fpaths = glob.glob(os.path.join(dir_path, "*.nc"))
fpath = fpaths[1]

# A bit slow to open ... just to let you know 
for fpath in fpaths: 
    ds = xr.open_dataset(fpath, chunks="auto")
    print(ds.attrs["northBoundLatitude"]) # lat 
    print(ds.attrs["eastBoundLongitude"]) # lon
    print(ds.attrs) # lon 


# Check all data has been downloaded and source info added to metadata !!!

# Disdrometer data Open Access of raw data and quality controlled (QC) data
# Risø - on the ground next to the tall met mast
# Raw: https://doi.org/10.11583/DTU.14501577.v1 
# QC: https://doi.org/10.11583/DTU.14501592
 
# Extract the same info for the others 

# Risø - on the top of the tall met mast
# Raw: https://doi.org/10.11583/DTU.14501598.v1 -_> Location 
# QC: https://doi.org/10.11583/DTU.14501601.v1


# Risø - on the ground next to the V52 met mast
# Raw: https://doi.org/10.11583/DTU.14501607.v1
# QC: https://doi.org/10.11583/DTU.14501610.v1


# Rødsand - offshore wind farm
# Raw: https://doi.org/10.11583/DTU.14501616.v1
# QC: https://doi.org/10.11583/DTU.14501619.v1


# Horns Rev 3 - offshore wind farm
# Raw: https://doi.org/10.11583/DTU.14501553.v1
# QC: https://doi.org/10.11583/DTU.14501568.v1


# Hvide Sande - DMI station
# Raw: https://doi.org/10.11583/DTU.14460006.v1
# QC: https://doi.org/10.11583/DTU.14501574.v1


# Thyborøn - DMI station
# Raw: https://doi.org/10.11583/DTU.14501622.v1
# QC: https://doi.org/10.11583/DTU.14501625.v1


# Voulund - DMI station
# Raw: https://doi.org/10.11583/DTU.14501628.v1
# QC: https://doi.org/10.11583/DTU.14501631.v1