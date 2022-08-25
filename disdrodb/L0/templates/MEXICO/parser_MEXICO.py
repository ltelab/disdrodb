#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:08:10 2022

@author: ghiggi
"""
import os
import glob
import xarray as xr

# Directory
from disdrodb.L0.io import (
    check_directories,
    get_campaign_name,
    create_directory_structure,
)

# Tools to develop the parser
from disdrodb.L0.template_tools import (
    check_column_names,
    infer_df_str_column_names,
    print_df_first_n_rows,
    print_df_random_n_rows,
    print_df_column_names,
    print_valid_L0_column_names,
    get_df_columns_unique_values_dict,
    print_df_columns_unique_values,
    print_df_summary_stats,
)

# L0A processing
from disdrodb.L0.L0A_processing import (
    read_raw_data,
    get_file_list,
    read_L0A_raw_file_list,
    cast_column_dtypes,
    write_df_to_parquet,  # TODO: add code to write to parquet a single file in 8.3 ... to check it works
)

# Metadata
from disdrodb.L0.metadata import read_metadata

# Standards
from disdrodb.L0.check_standards import check_sensor_name, check_L0A_column_names

# Logger
from disdrodb.utils.logger import create_logger

dir_path = "/ltenas3/0_Data/DISDRODB/TODO_Raw/MEXICO/OH_IIUNAM/data"
fpaths = glob.glob(os.path.join(dir_path, "*.nc"))
fpath = fpaths[0]

ds = xr.open_dataset(fpath)
