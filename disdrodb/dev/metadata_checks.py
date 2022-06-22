#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:42:56 2022

@author: ghiggi
"""
import os
import glob 
import yaml

def read_yaml(fpath):
    with open(fpath, "r") as f:
        attrs = yaml.safe_load(f)
    return attrs

def identify_missing_metadata(metadata_fpaths, keys):
    if isinstance(keys, str): 
        keys = [keys]
    for key in keys:
        for fpath in metadata_fpaths:
           #  print(fpath)
           metadata = read_yaml(fpath)
           if len(str(metadata.get(key,''))) == 0: # ensure is string to avoid error 
               print(f"Missing {key} at: ",fpath)
    return None 

def identify_missing_coords(metadata_fpaths):
    for fpath in metadata_fpaths:
       metadata = read_yaml(fpath)
       if metadata.get('longitude', -9999) == -9999 or metadata.get('latitude', -9999) == -9999:
           print(f"Missing lat lon coordinates at: {fpath}")
       elif metadata.get('longitude', -9999) > 180 or  metadata.get('longitude', -9999) < -180:  
           print("Unvalid longitude at : ",fpath)
       elif metadata.get('latitude', -9999) > 90 or metadata.get('latitude', -9999) < -90:
           print("Unvalid latitude at : ",fpath)
       else: 
           pass
    return None 

# # EXAMPLE
# ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Raw" # ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Processed"
# metadata_fpaths = glob.glob(os.path.join(ARCHIVE_DIR, "*/*/metadata/*.yml"))
# identify_missing_coords(metadata_fpaths)

# identify_missing_metadata(metadata_fpaths, keys="station_id") 
# identify_missing_metadata(metadata_fpaths, keys="station_name")
# identify_missing_metadata(metadata_fpaths, keys=["station_id","station_name"])
# identify_missing_metadata(metadata_fpaths, keys="campaign_name")

 #-----------------------------------------------------------------------------.
 # TODO 
 # - station_id 
 # - station_number --> to be replaced by station_id 
 # - station_name --> if missing, use station_id 
 
 # - TODO: define dtype of metadata !!!
 # - raise error if missing station_id
 # - raise error if missing campaign_name 
 #  
 

