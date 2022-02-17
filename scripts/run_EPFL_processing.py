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


parser_filepath = (
    "/home/ghiggi/Projects/disdrodb/disdrodb/readers/EPFL/parser_TICINO_2018.py"
)
# parser_TICINO_2018.py [OPTIONS] <raw_dir> <processed_dir>
cmd_options = "=".join([])
cmd = "".join(
    [
        "python",
        " ",
        parser_filepath,
        " ",
        "--l0_processing=",
        str(l0_processing),
        " ",
        "--l1_processing=",
        str(l1_processing),
        " ",
        "--write_zarr=",
        str(write_zarr),
        " ",
        "--write_netcdf=",
        str(write_netcdf),
        " ",
        "--force=",
        str(force),
        " ",
        "--verbose=",
        str(verbose),
        " ",
        "--debugging_mode=",
        str(debugging_mode),
        " ",
        "--lazy=",
        str(lazy),
        " ",
        raw_dir,
        " ",
        processed_dir,
    ]
)

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
