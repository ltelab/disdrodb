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

from disdrodb.dev.metadata_checks import ( 
    read_yaml,
    identify_missing_metadata,
    identify_missing_coords,
    )



ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Raw"
# ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Processed"
metadata_fpaths = glob.glob(os.path.join(ARCHIVE_DIR, "*/*/metadata/*.yml")) 

# metadata_fpaths
len(metadata_fpaths)

 #-----------------------------------------------------------------------------.
 # TODO 
 # - station_id 
 # - station_number --> to be replaced by station_id 
 # - station_name --> if missing, use station_id 
 # - raise error if missing campaign_name 
 # - raise error if missing station_id & station_name  
 # - Add station_name to all yaml 

identify_missing_metadata(metadata_fpaths, keys="campaign_name")
identify_missing_coords(metadata_fpaths)

#-----------------------------------------------------------------------------.
 
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

 


# import xarray as xr 
# ds = xr.open_dataset("/ltenas3/0_Data/DISDRODB/Raw/DELFT/WAGENINGEN/data/10/Disdrometer_20141001.nc")

#####-------------------------------------------------------------------------.
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
##############################################
#### Define station coordinate dataframes ####
##############################################
#### - Processed Data 
stations_dict = {}
for fpath in metadata_fpaths:
   metadata = read_yaml(fpath)
   lonlat =  metadata.get('longitude', -9999), metadata.get('latitude', -9999)
   # Deal with incomplete problem in DISDRODB
   try:   # TODO REMOVE
       fullname = get_station_full_name(metadata)
   except: 
       pass
   if lonlat[0] == -9999:
       print(fpath)
       continue
   stations_dict[fullname] = lonlat

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
#### - Gitter station coordinate for improved display 
def rand_jitter(arr):
    thr_degs = 0.1
    return arr + np.random.randn(len(arr)) * thr_degs

df_latlon['Lat'] = rand_jitter(df_latlon['Lat'])
df_latlon['Lon'] = rand_jitter(df_latlon['Lon'])

df_processed_latlon['Lat'] = rand_jitter(df_processed_latlon['Lat'])
df_processed_latlon['Lon'] = rand_jitter(df_processed_latlon['Lon'])

####--------------------------------------------------------------------------.
#### Display global coverage 
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy import crs as ccrs
alpha = 1
color_unprocessed = "black"
color_processed = "#006400" # DarkGreen
color_processed = "magenta"
marker = 'o'
crs_ref = ccrs.PlateCarree() # ccrs.AzimuthalEquidistant()
crs_proj = ccrs.Robinson() # ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw={'projection': crs_proj}, figsize=(18,12))
ax.set_global()
# ax.add_feature(cfeature.COASTLINE, edgecolor="black")
# ax.add_feature(cfeature.BORDERS, edgecolor="black")
# ax.gridlines()
ax.stock_img()

# - Plot unprocessed data
ax.scatter(x=df_latlon['Lon'], y=df_latlon['Lat'],
            color=color_unprocessed,
            edgecolor='None',
            marker=marker,
            alpha=alpha,
            transform=crs_ref) 

# - Plot processed data 
plt.scatter(x=df_processed_latlon['Lon'], y=df_processed_latlon['Lat'],
            color=color_processed,
            edgecolor='None',
            marker=marker,
            alpha=alpha,
            transform=crs_ref)  
 
plt.show()

#------------------------------------------------------------------------------.
### Display coverage per continent
marker = 'o'
alpha = 1
color_unprocessed = "black"
color_processed = "magenta"
continent_extent_dict = {  # [lon_start, lon_end, lat_start, lat_end].
    "Europe": [-10, 25, 36, 60],
    "CONUS": [-125,-74,12, 52], 
    "South America": [-81, -33, -52, 12]
}
continent_crs_proj_dict = {
    "Europe": ccrs.AlbersEqualArea(central_longitude=10, central_latitude=30, false_easting=0.0, false_northing=0.0, standard_parallels=(43.0, 62.0), globe=None),
    "CONUS": ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=40, false_easting=0.0, false_northing=0.0, standard_parallels=(20.0, 60.0), globe=None),
    "South America": ccrs.AlbersEqualArea(central_longitude=-60, central_latitude=-32, false_easting=0.0, false_northing=0.0, standard_parallels=(-5, -42), globe=None),
    }

for continent, extent in continent_extent_dict.items():
    crs_proj = continent_crs_proj_dict[continent]
    fig, ax = plt.subplots(subplot_kw={'projection': crs_proj}, figsize=(18,12))
    ax.set_extent(extent)
    # ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    # ax.add_feature(cfeature.BORDERS, edgecolor="black")
    # ax.gridlines()
    ax.stock_img()
    # - Plot unprocessed data
    plt.scatter(x=df_latlon['Lon'], y=df_latlon['Lat'],
                color=color_unprocessed,
                edgecolor='None',
                marker=marker, 
                alpha=alpha,
                transform=crs_ref) 
    # - Plot processed data 
    plt.scatter(x=df_processed_latlon['Lon'], y=df_processed_latlon['Lat'],
                color=color_processed,
                edgecolor='None',
                marker=marker,
                alpha=alpha,
                transform=crs_ref) ## Important
    plt.show()

# TODO: Add nicer background
# https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.geoaxes.GeoAxes.html#cartopy.mpl.geoaxes.GeoAxes.background_img

####--------------------------------------------------------------------------.
# TODO 
# - Total number of precipitation hours 
# - Longest record 
# --> Above per data source 