#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:18:04 2022

@author: ghiggi
"""
import os 
import sys 

# You need to set the disdrodb repo path in your .bashrc 
# export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Projects/disdrodb"

# Temporary in parser file I put the follow  [TO BE REMOVED]
import sys 
sys.path.append("/home/ghiggi/Projects/disdrodb")


# Set args 
raw_dir = "/home/ghiggi/Parsivel/TICINO_2018"
processed_dir = "/tmp/Processed/TICINO_2018"

l0_processing = True
l1_processing = True
force = True
verbose = True
debug_on = True
lazy = True
write_zarr = True
write_netcdf = True


parser_filepath = '/home/ghiggi/Projects/disdrodb/disdrodb/readers/EPFL/parser_TICINO_2018.py'
# parser_TICINO_2018.py [OPTIONS] <raw_dir> <processed_dir>
cmd_options = "=".join([ ])
cmd = "".join(["python", " ", 
               parser_filepath, " ", 
                "--l0_processing=", str(l0_processing), " ", 
                "--l1_processing=", str(l1_processing), " ", 
                "--write_zarr=", str(write_zarr), " ", 
                "--write_netcdf=", str(write_netcdf), " ", 
                "--force=", str(force), " ",  
                "--verbose=", str(verbose), " ",  
                "--debug_on=", str(debug_on), " ",  
                "--lazy=", str(lazy), " ",  
                raw_dir, " ",  
                processed_dir,  
                ])
                
os.system(cmd) 

#-----------------------------------------------------------------------------.
# TODO: 
# - Create dictionary with {CAMPAIGN_NAME: parser_*.py filepath}  
# - Loop over dictionary values (CAMPAIGN_NAME) 
# - Define cmd 
# - Run the command 

# --> Useful to test changes to code do not crash other parser
# --> debuggin_mode=True to speed up tests ;) 

#-----------------------------------------------------------------------------.


