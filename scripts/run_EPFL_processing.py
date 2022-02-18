#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:18:04 2022

@author: ghiggi
"""
import os
import subprocess

# You need to set the disdrodb repo path in your .bashrc
# export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Projects/disdrodb"

# -----------------------------------------------------------------------------.
EPFL_dict = {
    "PARSIVEL_2007": "parser_PARSIVEL_2007.py",
    "GENEPI_2007": "parser_GENEPI_2007.py",
    "EPFL_ROOF_2008V1": "parser_EPFL_ROOF_2008_V1.py",
    "EPFL_ROOF_2008V2": "parser_EPFL_ROOF_2012.py",
    "EPFL_ROOF_2011": "parser_EPFL_ROOF_2011.py",
    "EPFL_ROOF_2012": "parser_EPFL_ROOF_2008_V2.py",
    "EPFL_2009": "parser_EPFL_2009.py",
    "DAVOS_2009_2011": "parser_DAVOS_2009_2011.py",
    "HPICONET_2010": "parser_HPICONET_2010.py",
    "COMMON_2011": "parser_COMMON_2011.py",
    "RIETHOLZBACK_2011": "parser_RIETHOLZBACK_2011.py",
    "HYMEX_2012": "parser_HYMEX_2012.py",
    "PAYERNE_2014": "parser_PAYERNE_2014.py",
    "SAMOYLOV_2017_2019": "parser_SAMOYLOV_2017_2019.py",
    "TICINO_2018": "parser_TICINO_2018.py",
    "PLATO_2019": "parser_PLATO_2019.py",
}

parser_dir = "/ltenas3/0_Projects/disdrodb/disdrodb/readers/EPFL"
raw_dir = "/ltenas3/0_Data/ParsivelDB/EPFL/raw_data" 
processed_dir = "/ltenas3/0_Data/ParsivelDB/EPFL/raw_data" 
processed_dir = "/tmp/Processed/"

# Set args
raw_dir = "/home/ghiggi/Parsivel/TICINO_2018"
processed_dir = "/tmp/Processed/TICINO_2018"




l0_processing = True
l1_processing = True
force = True
verbose = True
debugging_mode = True
lazy = True
write_zarr = True
write_netcdf = True

for campaign_name in EPFL_dict.keys():
    parser_filepath = os.path.join(parser_dir, EPFL_dict[campaign_name])
)

get_parser_cmd(parser_filepath,
               raw_dir,
               processed_dir,
               l0_processing=True,
               l1_processing=True,
               write_zarr=False,
               write_netcdf=True,
               force=False,
               verbose=False,
               debugging_mode=False,
               lazy=True),
subprocess.run(cmd, shell=True)
# os.system(cmd)


# -----------------------------------------------------------------------------.
# TODO:
# - Create dictionary with {CAMPAIGN_NAME: parser_*.py filepath}
# - Loop over dictionary values (CAMPAIGN_NAME)
# - Define cmd
# - Run the command

# --> Useful to test changes to code do not crash other parser
# --> debuggin_mode=True to speed up tests ;)

# -----------------------------------------------------------------------------.
