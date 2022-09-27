#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:31:55 2022

@author: kimbo
"""
# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.

from disdrodb.L0 import run_L0


from disdrodb.L0.L0_processing import reader_generic_docstring, is_documented_by


@is_documented_by(reader_generic_docstring)
def reader(
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

    ####----------------------------------------------------------------------.
    ###########################
    #### CUSTOMIZABLE CODE ####
    ###########################
    #### - Define raw data headers
    # Notes
    # - In all files, the datalogger voltage hasn't the delimeter,
    #   so need to be split to obtain datalogger_voltage and rainfall_rate_32bit

    column_names = [
        "TO_BE_SPLITTED",
        "raw_drop_average_velocity",
        "raw_drop_number",
        "All_0",
    ]

    ##------------------------------------------------------------------------.
    #### - Define reader options

    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ",,"

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"

    # - Define reader engine
    #   - C engine is faster
    #   - Python engine is more feature-complete
    reader_kwargs["engine"] = "python"

    # - Define on-the-fly decompression of on-disk data
    #   - Available: gzip, bz2, zip
    reader_kwargs["compression"] = "gzip"

    # - Strings to recognize as NA/NaN and replace with standard NA flags
    #   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
    #                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
    #                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
    reader_kwargs["na_values"] = ["na", "", "error", "NA"]

    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    # Cast all to string
    reader_kwargs["dtype"] = str

    # Use for Nan value
    reader_kwargs["assume_missing"] = True

    ##------------------------------------------------------------------------.
    #### - Define facultative dataframe sanitizer function for L0 processing
    # - Enable to deal with bad raw data files
    # - Enable to standardize raw data files to L0 standards  (i.e. time to datetime)
    df_sanitizer_fun = None

    def df_sanitizer_fun(df, lazy=False):
        # Import dask or pandas
        if lazy:
            import dask.dataframe as dd
        else:
            import pandas as dd

        column_names = [
            "time",
            "temp",  # All nan values
            "temp1",  # Dataloger status
            "rainfall_rate_32bit",
            "rainfall_accumulated_32bit",
            "weather_code_synop_4680",
            "weather_code_synop_4677",
            "reflectivity_32bit",
            "mor_visibility",
            "sample_interval",
            "laser_amplitude",
            "number_particles",
            "sensor_heating_current",
            "sensor_battery_voltage",
            "sensor_status",
            "rainfall_amount_absolute_32bit",
            "temp2",  # Datalogger error
            "raw_drop_concentration",
            "raw_drop_average_velocity",
            "raw_drop_number",
        ]

        # Drop row if contains errors
        df = df[~df.TO_BE_SPLITTED.str.contains("Error in data reading! 0")]

        # Check again if file empty
        if len(df.index) == 0:
            raise ValueError("Error in all rows. The file has been skipped.")

        # Split TO_BE_SPLITTED
        df_to_parse = df["TO_BE_SPLITTED"].str.split(",", expand=True, n=20)

        # Concat togheter, avoid to save id, latitude and longitude and all_0
        if lazy:
            df = dd.concat(
                [df_to_parse.iloc[:, 3:], df.iloc[:, 1:-1]],
                axis=1,
                ignore_unknown_divisions=True,
            )
        else:
            df = dd.concat([df_to_parse.iloc[:, 3:], df.iloc[:, 1:-1]], axis=1)

        # Rename columns
        df.columns = column_names

        # Drop temp and all_nan
        df = df.drop(columns=["temp", "temp1", "temp2"])

        # - Convert time column to datetime
        df["time"] = dd.to_datetime(df["time"], format="%d-%m-%Y %H:%M:%S")

        # If raw_drop_number is nan, drop the row
        col_to_drop_if_na = [
            "raw_drop_concentration",
            "raw_drop_average_velocity",
            "raw_drop_number",
        ]
        df = df.dropna(subset=col_to_drop_if_na)

        # Drop rows with less than 4096 char on raw_drop_number
        df = df.loc[df["raw_drop_number"].astype(str).str.len() == 4095]

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    files_glob_pattern = "*.dat.gz*"

    ####----------------------------------------------------------------------.
    #### - Create L0 products
    run_L0(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        keep_l0a=keep_l0a,
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        lazy=lazy,
        single_netcdf=single_netcdf,
        # Custom arguments of the reader
        files_glob_pattern=files_glob_pattern,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
    )
