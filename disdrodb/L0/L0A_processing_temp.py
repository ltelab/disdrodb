#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:16:16 2023

@author: ghiggi
"""
# - specify only arguments: data_source, campaign_name, station_ids
# --> reader_name is taken from the yaml file !

# - ask for disdrodb_base_dir
# -->  instead of raw/processed_dir

# -----------------------------------------------------------------------------.
# TODO generate_l0_archive
# - search for yml file, search for stations with data, processes it
# - add station_ids --> reader, run_l0


# args
# def _run_disdrodb_l0a_station(disdrodb_dir, data_source, campaign_name, station):

#     run_reader(
#         data_source: str,
#         reader_name: str,
#         raw_dir: str,
#         processed_dir: str,
#         station_ids: list = None,
#         l0a_processing: bool = True,
#         l0b_processing: bool = True,
#         keep_l0a: bool = False,
#         force: bool = False,
#         verbose: bool = False,
#         debugging_mode: bool = False,
#         parallel: bool = True,
#         single_netcdf: bool = True,
#     )


disdrodb_dir = "/home/ghiggi/DISDRODB/"
disdrodb_dir = "/tmp/data/DISDRODB/"

data_sources = "EPFL"
data_sources = "EPFL"
campaign_names = "LOCARNO_2018"

list_info = available_stations(
    disdrodb_dir,
    product_level=product_level,
    data_sources=data_sources,
    campaign_names=campaign_names,
)
# Start the loop to launch the concatenation of each station
data_source, campaign_name, station = list_info[0]


# L0A requires raw dir
# L0B requires only processed dir

# disdrodb.L0.io currently expect processing of full archive
# --> station_ids
# --> be selective to station !!!


# Check directories and create directory structure
# TODO: flexible for station, type of processing

raw_dir, processed_dir = check_directories(raw_dir, processed_dir, force=force)
create_directory_structure(raw_dir, processed_dir)

# ----------------------------------------------------------------------------.

 
