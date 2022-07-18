#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:44:14 2022

@author: ghiggi
"""
import os
import glob 
import yaml
import matplotlib 
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy import crs as ccrs
from disdrodb.L0.check_metadata import ( 
    read_yaml,
    identify_missing_metadata,
    identify_missing_coords,
    )

# matplotlib.use('Agg') 



#-----------------------------------------------------------------------------. 
FIGS_DIR = "/home/ghiggi/Projects/disdrodb/figs"

ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Raw"
metadata_fpaths = glob.glob(os.path.join(ARCHIVE_DIR, "*/*/metadata/*.yml")) 

ARCHIVE1_DIR = "/ltenas3/0_Data/DISDRODB/TODO_Raw"
metadata1_fpaths = glob.glob(os.path.join(ARCHIVE1_DIR, "*/*/metadata/*.yml")) 

# ARCHIVE_DIR = "/ltenas3/0_Data/DISDRODB/Processed"
# metadata_fpaths = glob.glob(os.path.join(ARCHIVE1_DIR, "*/*/metadata/*.yml")) 

#-----------------------------------------------------------------------------.
# Check metadata
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
#### - Gitter station coordinate for improved display 
def rand_jitter(arr):
    thr_degs = 0.1
    return arr + np.random.randn(len(arr)) * thr_degs

df_latlon_jt = df_latlon.copy()
df_latlon_jt['Lat'] = rand_jitter(df_latlon['Lat'])
df_latlon_jt['Lon'] = rand_jitter(df_latlon['Lon'])

df_processed_latlon_jt = df_processed_latlon.copy()
df_processed_latlon_jt['Lat'] = rand_jitter(df_processed_latlon['Lat'])
df_processed_latlon_jt['Lon'] = rand_jitter(df_processed_latlon['Lon'])

####--------------------------------------------------------------------------.
#### Display global coverage with low resolution 
figsize = (18,12)
dpi = 800
dpi = 500
alpha = 1
color_unprocessed = "black"
color_processed = "#006400" # DarkGreen
color_processed = "magenta"
marker = 'o'
marker_size = 5
crs_ref = ccrs.PlateCarree() # ccrs.AzimuthalEquidistant()
crs_proj = ccrs.Robinson() # ccrs.PlateCarree()


fig, ax = plt.subplots(subplot_kw={'projection': crs_proj}, figsize=figsize, dpi=dpi)
ax.set_global()
ax.stock_img()

# - Plot unprocessed data
ax.scatter(x=df_latlon_jt['Lon'],
           y=df_latlon_jt['Lat'],
            color=color_unprocessed,
            edgecolor='None',
            marker=marker,
            s = marker_size, 
            alpha=alpha,
            transform=crs_ref) 

# - Plot processed data 
ax.scatter(x=df_processed_latlon_jt['Lon'], 
           y=df_processed_latlon_jt['Lat'],
           color=color_processed,
           edgecolor='None',
           marker=marker,
           s = marker_size, 
           alpha=alpha,
           transform=crs_ref)  
 
fig.savefig(os.path.join(FIGS_DIR, "global_low_res.png"))

####--------------------------------------------------------------------------.
#### Display global coverage with high resolution 


background_fpath = "/ltenas3/0_GIS/Backgrounds/Robinson/HYP_50M_SR_W/HYP_50M_SR_W.tif"
ds_reproj = xr.open_dataset(background_fpath, engine="rasterio") 
da_reproj = ds_reproj["band_data"]

# Ensure image is converted to 0-1 
da_reproj = da_reproj/255.0

# Define projection crs
crs_proj = ccrs.Robinson()
extent = (crs_proj.x_limits[0], crs_proj.x_limits[1], crs_proj.y_limits[0], crs_proj.y_limits[1]) 
 
figsize = (20,10)
dpi = 800
# dpi = 500
alpha = 1
color_unprocessed = "black"
color_processed = "#006400"  # DarkGreen
color_processed = "#magenta" # DarkGreen
color_processed = "#f06609"  # dark orange      
marker = 'o'#
marker_size = 4
 


fig, ax = plt.subplots(subplot_kw={'projection': crs_proj}, figsize=figsize, dpi=dpi)
# - Set global 
ax.set_global()
# - Add background  
ax.imshow(da_reproj.transpose('y','x',...).data,
          origin='upper',
          interpolation = "nearest",
          transform=crs_proj,
          extent=extent,
          )
# - Plot unprocessed data
ax.scatter(x=df_latlon_jt['Lon'],
           y=df_latlon_jt['Lat'],
            color=color_unprocessed,
            edgecolor='None',
            marker=marker,
            s = marker_size, 
            alpha=alpha,
            transform=ccrs.PlateCarree()) 

# - Plot processed data 
ax.scatter(x=df_processed_latlon_jt['Lon'], 
           y=df_processed_latlon_jt['Lat'],
           color=color_processed,
           edgecolor='None',
           marker=marker,
           s = marker_size, 
           alpha=alpha,
           transform=ccrs.PlateCarree())  
 
fig.savefig(os.path.join(FIGS_DIR, "global_map_hres.png"))



####--------------------------------------------------------------------------.
#### Display coverage per continent
crs_ref = ccrs.PlateeCarree()
marker = 'o'
marker_size = 5
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
    fig, ax = plt.subplots(subplot_kw={'projection': crs_proj}, figsize=(18,12), dpi=dpi)
    ax.set_extent(extent)
    ax.stock_img()
    
    # - Plot unprocessed data
    ax.scatter(x=df_latlon_jt['Lon'],
               y=df_latlon_jt['Lat'],
                color=color_unprocessed,
                edgecolor='None',
                marker=marker, 
                s = marker_size, 
                alpha=alpha,
                transform=crs_ref) 
    
    # - Plot processed data 
    ax.scatter(x=df_processed_latlon_jt['Lon'],
               y=df_processed_latlon_jt['Lat'],
                color=color_processed,
                edgecolor='None',
                marker=marker,
                s = marker_size, 
                alpha=alpha,
                transform=crs_ref) ## Important
    fig.savefig(os.path.join(FIGS_DIR, continent + "_map.png"))
    
####--------------------------------------------------------------------------.
