#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:20:46 2022

@author: ghiggi
"""

# TODO: adapt to new readers !!!!


def get_reader_cmd(
    reader_filepath,
    raw_dir,
    processed_dir,
    l0a_processing=True,
    l0b_processing=True,
    keep_l0a=False,
    force=False,
    verbose=False,
    debugging_mode=False,
    lazy=True,
    single_netcdf=True,
):
    """Create command to launch reader processing from Terminal."""
    # reader_TICINO_2018.py [OPTIONS] <raw_dir> <processed_dir>
    cmd = "".join(
        [
            "python",
            " ",
            reader_filepath,
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
