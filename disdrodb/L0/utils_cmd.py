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
    l0_processing=True,
    l1_processing=True,
    # write_zarr=False,
    write_netcdf=True,
    force=False,
    verbose=False,
    debugging_mode=False,
    lazy=True,
):
    """Create command to launch parser processing from Terminal."""
    # parser_TICINO_2018.py [OPTIONS] <raw_dir> <processed_dir>
    cmd = "".join(
        [
            "python",
            " ",
            parser_filepath,
            " ",
            "--L0A_processing=",
            str(l0_processing),
            " ",
            "--L0B_processing=",
            str(l1_processing),
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
    return cmd
