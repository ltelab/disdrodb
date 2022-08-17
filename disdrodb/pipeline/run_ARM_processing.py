#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:38:37 2022

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
# ARM_ld --> OTT Parsivel 2 
# ARM_lpm --> Thies LPM 
campaign_dict = {
    # "ALASKA": "parser_ARM_lpm.py",
    "ACE_ENA": "parser_ARM_ld.py",
    "AWARE": "parser_ARM_ld.py",
    "CACTI": "parser_ARM_ld.py",
    "COMBLE": "parser_ARM_ld.py",
    "GOAMAZON": "parser_ARM_ld.py",
    "MARCUS": "parser_ARM_ld.py", # MARCUS S1, MARCUS S2 are mobile ...
    "MICRE": "parser_ARM_ld.py",
    "MOSAIC": "parser_ARM_ld.py", # MOSAIC M1, MOSAIC S3 are mobile ...
    "SAIL": "parser_ARM_ld.py",
    "SGP": "parser_ARM_ld.py",
    "TRACER": "parser_ARM_ld.py",
    
}     
       
#### Define filepaths
parser_dir = "/ltenas3/0_Projects/disdrodb/disdrodb/readers/ARM" # TO CHANGE
raw_base_dir = "/ltenas3/0_Data/DISDRODB/Raw/ARM"
processed_base_dir = "/ltenas3/0_Data/DISDRODB/Processed/ARM"
# processed_base_dir = "/tmp/DISDRODB/ARM"

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

    print(subprocess.run(cmd, shell=True))
    os.system(cmd)

# -----------------------------------------------------------------------------.
# TODO:
# --> Useful to test changes to code do not crash other parser
# --> debuggin_mode=True to speed up tests ;)

# -----------------------------------------------------------------------------.