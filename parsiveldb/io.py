#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:10:59 2021

@author: ghiggi
"""

def _write_to_parquet(df, fpath, force=False):    
    # TODO: schema args, and vompress options, chunks 
    # TODO: If force=False and dir exists, raise Error. If True remove and write 
    try:
        df.to_parquet(fpath , schema='infer')
    except (Exception) as e:
        raise ValueError("Can not convert to parquet file. The error is {}".format(e))
     
## Kimbo
# - correct header names
# - dtype, attrs standards 
# - Check folder exists if force=True, 
# - coordinate standards ? 
#   get_velocity_bin_center(): 
    # - args: instrument=Parsivel,Thies .. if, elif 
# - click https://click.palletsprojects.com/en/8.0.x/ -> default type=bool



### GG
## Netcdf encoding 
## Parquet schema 