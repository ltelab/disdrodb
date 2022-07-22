#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:20:46 2022

@author: ghiggi
"""

# TODO: adapt to new readers !!!!

def get_parser_cmd(
    parser_filepath,
    raw_dir,
    processed_dir,
    l0a_processing,
    l0b_processing,
    keep_l0a,
    force,
    verbose,
    debugging_mode,
    lazy,
    single_netcdf, 
):
    """Create command to launch parser processing from Terminal."""
    # parser_TICINO_2018.py [OPTIONS] <raw_dir> <processed_dir>
    cmd = "".join(
        [
            "python",
            " ",
            parser_filepath,
            " ",
            "--l0a_processing=",
            str(l0a_processing),
            " ",
            "--l0b_processing=",
            str(l0b_processing),
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
            "--keep_l0a=",
            str(keep_l0a),
            " ",
            "--single_netcdf=",
            str(single_netcdf),
            " ",
            raw_dir,
            " ",
            processed_dir,
        ]
    )
    return cmd
