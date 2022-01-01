#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 18:30:31 2022

@author: ghiggi
"""
#### Conventions 
# <campaign_name> always UPPER CASE 
# Set directory also upper case

# A yaml file for each station_id 

#### Folder structure 
# raw_dir:
# .../campaign_name
# - /data/<station_id>
# - /metadata

# processed_dir 
# .../processed/<campaign_name>
# - /L0
# - /L1 (TODO: L1_NETCDF, L1_ZARR)
# - /info 
# - /metadata


### Recommended steps to develop a parser 
# - Copy template_parser_dev.py
# - Start reading a single file (lazy=False)
# - Infer headers and reader kwargs 
# - Check it works also if lazy=True (perform df.compute() to see the df) 

# - Implement a parser file copying and modifiing the template_parser.py a
# - Add the parser to disdrodb/readers  
# - Try running the parser in debugging_mode=True with both lazy=True and lazy=False 
# - Try running the parser in debugging_mode=False with both lazy=True 

# - Look at the L0 file and check it looks correct 

## Pay attention to: 
# TODO KIMBO 
