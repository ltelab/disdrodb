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


ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Raw"
# ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Processed"
metadata_fpaths = glob.glob(os.path.join(ARCHIVE_DIR, "*/*/metadata/*.yml")) 

# metadata_fpaths = glob.glob(os.path.join(ARCHIVE_DIR, "*/*/*/metadata/*.yml"))   

 
metadata_fpaths

def read_yaml(fpath):
    with open(fpath, "r") as f:
        attrs = yaml.safe_load(f)
    return attrs

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

def identify_missing_metadata(metadata_fpaths, keys):
    if isinstance(keys, str): 
        keys = [keys]
    for key in keys:
        for fpath in metadata_fpaths:
           metadata = read_yaml(fpath)
           if len(metadata.get(key,'')) == 0:
               print(f"Missing {key} at: ",fpath)
    return None 

def identify_missing_coords(metadata_fpaths):
    for fpath in metadata_fpaths:
       metadata = read_yaml(fpath)
       if metadata.get('longitude', -9999) == -9999 or metadata.get('latitude', -9999) == -9999:
           print(f"Missing lat lon coordinates at: ",fpath)
       elif metadata.get('longitude', -9999) > 180 or  metadata.get('longitude', -9999) < -180:  
           print(f"Unvalid longitude at : ",fpath)
       elif metadata.get('latitude', -9999) > 90 or metadata.get('latitude', -9999) < -90:
           print(f"Unvalid latitude at : ",fpath)
       else: 
           pass
    return None 

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
print(sensor_name_dict)

sensors, counts = np.unique(list(sensor_name_dict.values()), return_counts=True)
sensors_stats = dict(zip(sensors, counts))
sensors_stats = dict(sorted(sensors_stats.items(), key=lambda x: x[1], reverse=True))
print(sensors_stats)

####--------------------------------------------------------------------------.
############################### 
#### Plot spatial coverage ####
############################### 
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
       continue
   stations_dict[fullname] = lonlat

# Check not overwriting keys ... (equal full name)
# assert len(stations_dict) == len(metadata_fpaths)  # TODO REMOVE

lons =  [t[0] for t in stations_dict.values()]
lats =  [t[1] for t in stations_dict.values()]  
gdf = gpd.GeoDataFrame(stations_dict.keys(), 
                       geometry=gpd.points_from_xy(lons, lats))

# gdf.plot()

#------------------------------------------------------------------------------.
### Display global coverage 
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy import crs as ccrs
crs_ref = ccrs.PlateCarree() # ccrs.AzimuthalEquidistant()
crs_proj = ccrs.Robinson() # ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': crs_proj})
ax.set_global()
# ax.add_feature(cfeature.COASTLINE, edgecolor="black")
# ax.add_feature(cfeature.BORDERS, edgecolor="black")
ax.gridlines()
ax.stock_img()

plt.scatter(x=gdf['geometry'].x, y=gdf['geometry'].y,
            color="black",
            s=2,
            alpha=0.5,
            transform=crs_ref) ## Important

plt.show()

#------------------------------------------------------------------------------.
### Display coverage per continent 
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
    fig, ax = plt.subplots(subplot_kw={'projection': crs_proj})
    ax.set_extent(extent)
    # ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    # ax.add_feature(cfeature.BORDERS, edgecolor="black")
    # ax.gridlines()
    ax.stock_img()
    plt.scatter(x=gdf['geometry'].x, y=gdf['geometry'].y,
                color="black",
                s=1,
                alpha=0.5,
                transform=crs_ref) ## Important
    plt.show()

# TODO: Add nicer background
# https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.geoaxes.GeoAxes.html#cartopy.mpl.geoaxes.GeoAxes.background_img

####--------------------------------------------------------------------------.