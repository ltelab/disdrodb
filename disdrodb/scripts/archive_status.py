#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:41:19 2022

@author: ghiggi
"""
import os
import glob 
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib 

from disdrodb.L0.check_metadata import ( 
    read_yaml,
    identify_missing_metadata,
    identify_missing_coords,
    )

# EPFL, NCAR, ARM, NASA, NERC/MetOffice, DELFT, DTU, UGA-IGE,  OH-IIUNAM, CEMADEN, GID,
FIGS_DIR = "/home/ghiggi/Projects/disdrodb/figs"

ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Raw"
metadata_fpaths = glob.glob(os.path.join(ARCHIVE_DIR, "*/*/metadata/*.yml")) 

ARCHIVE1_DIR = "/ltenas3/0_Data/DISDRODB/TODO_Raw"
metadata1_fpaths = glob.glob(os.path.join(ARCHIVE1_DIR, "*/*/metadata/*.yml")) 
# ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Processed"

####--------------------------------------------------------------------------.
identify_missing_metadata(metadata_fpaths, keys="campaign_name")
identify_missing_coords(metadata_fpaths)
identify_missing_coords(metadata1_fpaths)

####--------------------------------------------------------------------------.
##############################################
#### Define station coordinate dataframes ####
##############################################
def get_station_full_name(metadata): 
    campaign_name = metadata.get("campaign_name",'') 
    station_name =  metadata.get("station_name",'')
    station_id =  metadata.get("station_id",'') 
    # Check campagn name is specified 
    if len(campaign_name) == 0: 
        raise ValueError("The campaign_name must be specified in the metadata.")
        
    # Check that either station name and station_id are specified 
    if len(station_name) == 0 and len(station_id) == 0: 
        raise ValueError("Either station_id or station_name must be specified in the metadata.")
    
    # Define full name 
    if len(station_name) == 0: 
        fullname = campaign_name + ": Station " +  station_id
    elif len(station_id) == 0: 
        fullname = campaign_name + ": " +  station_name 
    else: 
        fullname = campaign_name + ": " +  station_name + " (S" + station_id + ")"
    return fullname

def get_stations_dict(metadata_fpaths):
    stations_dict = {}
    for fpath in metadata_fpaths:
        metadata = read_yaml(fpath)
        lonlat =  metadata.get('longitude', -9999), metadata.get('latitude', -9999)
        # Deal with incomplete problem in DISDRODB
        try:   # TODO REMOVE
            fullname = get_station_full_name(metadata)
        except: 
            pass
        if lonlat[0] == -9999 or isinstance(lonlat[0], type(None)):
            print(fpath)
            continue
        stations_dict[fullname] = lonlat
    return stations_dict

#### - Retrieve archived data 
stations_dict = get_stations_dict(metadata_fpaths)
stations1_dict = get_stations_dict(metadata1_fpaths)

stations_dict.update(stations1_dict)

# Check not overwriting keys ... (equal full name)
# assert len(stations_dict) == len(metadata_fpaths)  # TODO REMOVE

lons =  [t[0] for t in stations_dict.values()]
lats =  [t[1] for t in stations_dict.values()]  
 
df_processed_latlon = pd.DataFrame({"Lat": lats, "Lon": lons})

####--------------------------------------------------------------------------.
#### - Literature data 
### Literature list 
literature_table_fpath = "/ltenas3/0_Data/DISDRODB/DISDRO_List.xlsx"
df_list = pd.read_excel(literature_table_fpath)
df_latlon = df_list[['Lat','Lon']]
df_latlon = df_latlon[df_latlon['Lat'].notnull()]

### IGE list 
ige_network_fpath = "/ltenas3/0_Data/DISDRODB/TODO_Raw/FRANCE/IGE/IGE_DSD_locations_v2.xlsx"
df_ige = pd.read_excel(ige_network_fpath)
df_ige_latlon = df_ige[['Latitude','Longitude']]
df_ige_latlon.columns = ['Lat','Lon']
df_processed_latlon = df_processed_latlon.append(df_ige_latlon)

####--------------------------------------------------------------------------.
### - Export to shapefile 
import geopandas as gpd 
gdf = gpd.GeoDataFrame(
    df_processed_latlon, geometry=gpd.points_from_xy(df_processed_latlon['Lon'], df_processed_latlon['Lat']))    
gdf.to_file("/home/ghiggi/processed.shp")

gdf = gpd.GeoDataFrame(
    df_latlon, geometry=gpd.points_from_xy(df_latlon['Lon'], df_latlon['Lat']))    
gdf.to_file("/home/ghiggi/listed.shp")
 
####--------------------------------------------------------------------------.
################################# 
#### Data Sources statistics ####
#################################
def get_data_source_from_metadata_yml(fpath): 
    data_source = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fpath))))  
    return data_source 

list_data_source = []
for fpath in metadata_fpaths:
   data_source = get_data_source_from_metadata_yml(fpath)
   list_data_source.append(data_source)
 
data_sources, counts = np.unique(list_data_source, return_counts=True)
data_sources_stats = dict(zip(data_sources, counts))
data_sources_stats = dict(sorted(data_sources_stats.items(), key=lambda x: x[1], reverse=True))
print(data_sources_stats)

####--------------------------------------------------------------------------.
################################ 
#### Sensor type statistics ####
################################
sensor_name_dict = {}
for fpath in metadata_fpaths:
   metadata = read_yaml(fpath)
   sensor_name =  metadata.get('sensor_name','') 
   # Deal with incomplete problem in DISDRODB
   try:   # TODO REMOVE
       fullname = get_station_full_name(metadata)
   except: 
       pass
   sensor_name_dict[fullname] = sensor_name
   
# print(sensor_name_dict)

sensors, counts = np.unique(list(sensor_name_dict.values()), return_counts=True)
sensors_stats = dict(zip(sensors, counts))
sensors_stats = dict(sorted(sensors_stats.items(), key=lambda x: x[1], reverse=True))
print(sensors_stats)

####--------------------------------------------------------------------------.
# TODO 
# - Total number of precipitation hours 
# - Longest record 
# --> Above per data source 