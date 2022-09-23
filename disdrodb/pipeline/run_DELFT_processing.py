#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:18:04 2022

@author: ghiggi
"""
import os
import subprocess
from disdrodb.pipeline.utils_cmd import get_reader_cmd
from pathlib import Path

# You need to set the disdrodb repo path in your .bashrc
# export PYTHONPATH="${PYTHONPATH}:/home/sguzzo/Projects/disdrodb"
# You need to activate the disdrodb environment: conda activate disdrodb

# -----------------------------------------------------------------------------.
#### Define campaign dictionary
DELFT_dict = {
    "CABAUW": "reader_RASPBERRY.py",
}

#### Define filepaths
home = str(Path.home())
reader_dir = "/home/sguzzo/PycharmProjects/disdrodb/disdrodb/readers/DELFT"
raw_base_dir = "/home/sguzzo/Parsivel/RAW_TELEGRAM"
processed_base_dir = "/tmp/Processed"
# processed_base_dir = "/tmp/Processed/DELFT"

#### Processing settings
l0a_processing = True
l0b_processing = True
force = True
verbose = True
debugging_mode = False  # da cambiare se vuoi processare piu dei primi 5 files
lazy = False
# write_zarr = False
keep_l0a = True
single_netcdf = True

#### Process all campaigns
for campaign_name in DELFT_dict.keys():
    reader_filepath = os.path.join(reader_dir, DELFT_dict[campaign_name])
    cmd = get_reader_cmd(
        reader_filepath=reader_filepath,
        raw_dir=os.path.join(raw_base_dir, campaign_name),
        processed_dir=os.path.join(processed_base_dir, campaign_name),
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        keep_l0a=keep_l0a,
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        lazy=lazy,
        single_netcdf=single_netcdf,
    )

    subprocess.run(cmd, shell=True)
    # os.system(cmd)

# -----------------------------------------------------------------------------.
# TODO:
# --> Useful to test changes to code do not crash other reader
# --> debugging_mode=True to speed up tests ;)

# -----------------------------------------------------------------------------.
