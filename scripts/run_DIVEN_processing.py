#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:03:10 2022

@author: ghiggi
"""
import os
import subprocess
from disdrodb.utils.parser import get_parser_cmd

# You need to set the disdrodb repo path in your .bashrc
# export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Projects/disdrodb"
# You need to activate the disdrodb envirnment: conda activate disdrodb

# -----------------------------------------------------------------------------.
#### Define campaign dictionary
campaign_dict = {
    "DIVEN": "parser_DIVEN.py",
}     
       
#### Define filepaths
parser_dir = "/ltenas3/0_Projects/disdrodb/disdrodb/readers/DIVEN" # TO CHANGE
raw_base_dir = "/ltenas3/0_Data/DISDRODB/Raw/UK"
processed_base_dir = "/ltenas3/0_Data/DISDRODB/Processed/UK"
# processed_base_dir = "/tmp/DISDRODB/UK"

#### Processing settings
force = True
verbose = True
debugging_mode = False
lazy = True

#### Process all campaigns
campaign_name = list(campaign_dict.keys())[0]
for campaign_name in campaign_dict.keys():
    print("Processing: ", campaign_name)
    parser_filepath = os.path.join(parser_dir, campaign_dict[campaign_name])
    raw_dir = os.path.join(raw_base_dir, campaign_name)
    processed_dir = os.path.join(processed_base_dir, campaign_name)
    cmd = get_parser_cmd(
        parser_filepath=parser_filepath,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        lazy=lazy,
    )
    
    subprocess.run(cmd, shell=True)
    # os.system(cmd)

# -----------------------------------------------------------------------------.
# TODO:
# --> Useful to test changes to code do not crash other parser
# --> debuggin_mode=True to speed up tests ;)

# -----------------------------------------------------------------------------.