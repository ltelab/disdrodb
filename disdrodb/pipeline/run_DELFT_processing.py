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
l0_processing = True
l1_processing = True
force = True
verbose = True
debugging_mode = False  # da cambiare se vuoi processare piu dei primi 5 files
lazy = False
# write_zarr = False
write_netcdf = True

#### Process all campaigns
for campaign_name in DELFT_dict.keys():
    reader_filepath = os.path.join(reader_dir, DELFT_dict[campaign_name])
    cmd = get_reader_cmd(
        reader_filepath=reader_filepath,
        raw_dir=os.path.join(raw_base_dir, campaign_name),
        processed_dir=os.path.join(processed_base_dir, campaign_name),
        l0_processing=l0_processing,
        l1_processing=l1_processing,
        write_netcdf=write_netcdf,
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        lazy=lazy,
    )

    subprocess.run(cmd, shell=True)
    # os.system(cmd)

# -----------------------------------------------------------------------------.
# TODO:
# --> Useful to test changes to code do not crash other reader
# --> debugging_mode=True to speed up tests ;)

# -----------------------------------------------------------------------------.
