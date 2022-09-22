#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:18:04 2022

@author: ghiggi
"""
import os
import subprocess
from disdrodb.pipeline.utils_cmd import get_reader_cmd

# You need to set the disdrodb repo path in your .bashrc
# export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Projects/disdrodb"
# You need to activate the disdrodb envirnment: conda activate disdrodb

# -----------------------------------------------------------------------------.
#### Define campaign dictionary
EPFL_dict = {
    "LOCARNO_2018": "reader_LOCARNO_2018.py",
    "PARSIVEL_2007": "reader_PARSIVEL_2007.py",
    "GENEPI_2007": "reader_GENEPI_2007.py",  # Asked to discard into metadata
    "EPFL_ROOF_2008_1": "reader_EPFL_ROOF_2008_1.py",
    "EPFL_ROOF_2008_2": "reader_EPFL_ROOF_2008_2.py",
    "EPFL_ROOF_2011": "reader_EPFL_ROOF_2011.py",
    "EPFL_ROOF_2012": "reader_EPFL_ROOF_2012.py",
    "EPFL_2009": "reader_EPFL_2009.py",
    "DAVOS_2009_2011": "reader_DAVOS_2009_2011.py",
    "HPICONET_2010": "reader_HPICONET_2010.py",
    "COMMON_2011": "reader_COMMON_2011.py",
    "RIETHOLZBACK_2011": "reader_RIETHOLZBACK_2011.py",
    "HYMEX": "reader_HYMEX.py",
    "HYMEX": "reader_HYMEX.py",
    "PARADISO_2014": "reader_PARADISO_2014.py",
    "SAMOYLOV_2017_2019": "reader_SAMOYLOV_2017_2019.py",
    "LOCARNO_2018": "reader_LOCARNO_2018.py",
    "PLATO_2019": "reader_PLATO_2019.py",
    "DAVOS_2009": "reader_DAVOS_2009.py",
    "EPFL_ROOF_2010": "reader_EPFL_ROOF_2010.py",
    "PARADISO_2014": "reader_PARADISO_2014.py",
    "RACLETS_2019": "reader_RACLETS_2019.py",
    "SAMOYLOV_2017_2019": "reader_SAMOYLOV_2017_2019.py",
}

#### Define filepaths
reader_dir = "/ltenas3/0_Projects/disdrodb/disdrodb/readers/EPFL"  # TODO: this should change to the current package
reader_dir = "~/disdrodb/disdrodb/L0/readers/EPFL"

raw_base_dir = "/ltenas3/0_Data/DISDRODB/Raw/EPFL"

processed_base_dir = "/ltenas3/0_Data/DISDRODB/Processed/EPFL"
processed_base_dir = "/tmp/DISDRODB/Processed/EPFL"
processed_base_dir = "/home/kcandolf/tmp/DISDRODB/Processed/EPFL"


#### Processing settings
l0a_processing = True
l0b_processing = True
keep_l0a = True
force = True
verbose = True
debugging_mode = True
lazy = True
single_netcdf = True


#### Process all campaigns
for campaign_name in EPFL_dict.keys():
    reader_filepath = os.path.join(reader_dir, EPFL_dict[campaign_name])
    cmd = get_reader_cmd(
        reader_filepath=reader_filepath,
        raw_dir=os.path.join(raw_base_dir, campaign_name),
        processed_dir=os.path.join(processed_base_dir, campaign_name),
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        keep_l0a=keep_l0a,
        single_netcdf=single_netcdf,
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
# --> debuggin_mode=True to speed up tests ;)

# -----------------------------------------------------------------------------.
