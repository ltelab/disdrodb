#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    ##------------------------------------------------------------------------.
    #### - Define column names
    column_names = ["TO_SPLIT"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False
    # Skip first row as columns names
    reader_kwargs["header"] = None
    # Skip the first row (header)
    reader_kwargs["skiprows"] = 1
    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"
    # Define encoding
    reader_kwargs["encoding"] = "latin1"
    # - Define reader engine
    #   - C engine is faster
    #   - Python engine is more feature-complete
    reader_kwargs["engine"] = "python"
    # - Define on-the-fly decompression of on-disk data
    #   - Available: gzip, bz2, zip
    reader_kwargs["compression"] = "infer"
    # - Strings to recognize as NA/NaN and replace with standard NA flags
    #   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
    #                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
    #                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
    reader_kwargs["na_values"] = ["na", "", "error"]
    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    def df_sanitizer_fun(df, lazy=False):
        # Import dask or pandas
        if lazy:
            import dask.dataframe as dd
        else:
            import pandas as dd

        # The delimiter ; is used for separating both the variables and the
        #   values of the raw spectrum. So we need to retrieve the columns
        #   inside the sanitizer assuming a fixed number of columns.
        df = df["TO_SPLIT"].str.split(";", expand=True, n=16)

        # Define the column names
        column_names = [
            "date",
            "time",
            "rainfall_rate_32bit",
            "rainfall_accumulated_32bit",
            "reflectivity_32bit",
            "mor_visibility",
            "laser_amplitude",
            "number_particles",
            "sensor_temperature",
            "sensor_heating_current",
            "sensor_battery_voltage",
            "rain_kinetic_energy",
            "snowfall_rate",
            "weather_code_synop_4680",
            "weather_code_metar_4678",
            "weather_code_nws",
            "raw_drop_number",
        ]
        df.columns = column_names

        # Define the time column
        df["time"] = df["date"] + "-" + df["time"]
        df["time"] = dd.to_datetime(df["time"], format="%Y/%m/%d-%H:%M:%S")
        df = df.drop(columns=["date"])

        # Preprocess the raw spectrum
        # - The '<SPECTRUM>ZERO</SPECTRUM>'  indicates no drops detected
        # - So replace the string with '' so that L0B processing generate a matrix filled by 0s.
        df["raw_drop_number"] = df["raw_drop_number"].str.replace(
            "<SPECTRUM>ZERO</SPECTRUM>", "''"
        )
        # Remove <SPECTRUM> and </SPECTRUM>" acroynms from the raw_drop_number field
        df["raw_drop_number"] = df["raw_drop_number"].str.replace("<SPECTRUM>", "")
        df["raw_drop_number"] = df["raw_drop_number"].str.replace("</SPECTRUM>", "")

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_id>
    files_glob_pattern = "*.txt"  # There is only one file without extension

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
        lazy=False,  # The actual solution work only with pandas
        single_netcdf=single_netcdf,
        # Custom arguments of the reader
        files_glob_pattern=files_glob_pattern,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
    )
