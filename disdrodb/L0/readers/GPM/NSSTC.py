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
    column_names = ["time", "TO_BE_PARSED"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Need for zipped raw file (GPM files)
    reader_kwargs["zipped"] = True
    # - Define delimiter
    reader_kwargs["delimiter"] = ";"
    # - Skip first row as columns names
    reader_kwargs["header"] = None
    # - Skip file with encoding errors
    reader_kwargs["encoding_errors"] = "ignore"
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
    reader_kwargs["compression"] = "infer"
    # - Strings to recognize as NA/NaN and replace with standard NA flags
    #   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
    #                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
    #                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
    reader_kwargs["na_values"] = ["na", "", "error", "-.-"]
    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    # Deal with changing file format after 25 feb 2011 by the documentation
    # - https://ghrc.nsstc.nasa.gov/pub/fieldCampaigns/gpmValidation/relatedProjects/nsstc/parsivel/doc/gpm_parsivel_nsstc_dataset.html).
    # - TODO: code below might exploit df['time'] instead of n_delimiters

    def df_sanitizer_fun(df, lazy=False):
        # - Import dask or pandas
        if lazy:
            import dask.dataframe as dd
        else:
            import pandas as dd

        # - Check 'time' string length
        # --> Enable to detect rows which are corrupted
        df = df[df["time"].str.len() == 14]

        # - Convert time column to datetime
        df["time"] = dd.to_datetime(df["time"], format="%Y%m%d%H%M%S")

        # Compute number of delimiters in the column to be parsed
        # - Count commas on the first row to determine the columns number
        n_delimiters = df["TO_BE_PARSED"].iloc[0].count(",").item()

        if n_delimiters == 1027:
            # - Select valid rows
            df = df.loc[df["TO_BE_PARSED"].str.count(",") == 1027]
            # - Get time column
            df_time = df["time"]
            # - Split the column be parsed
            df = df["TO_BE_PARSED"].str.split(",", expand=True, n=3)
            # - Assign column names
            column_names = [
                "station_name",
                "sensor_status",
                "sensor_temperature",
                "raw_drop_number",
            ]
            df.columns = column_names
            # - Add missing columns with NAN value
            df["number_particles"] = "NaN"
            df["rainfall_rate_32bit"] = "NaN"
            df["reflectivity_32bit"] = "NaN"
            df["mor_visibility"] = "NaN"
            df["weather_code_synop_4680"] = "NaN"
            df["weather_code_synop_4677"] = "NaN"
            # - Add time column
            df["time"] = df_time

        elif n_delimiters == 1033:
            # - Select valid rows
            df = df.loc[df["TO_BE_PARSED"].str.count(",") == 1033]
            # - Get time column
            df_time = df["time"]
            # - Split the column be parsed
            df = df["TO_BE_PARSED"].str.split(",", expand=True, n=9)
            # - Assign column names
            column_names = [
                "station_name",
                "sensor_status",
                "sensor_temperature",
                "number_particles",
                "rainfall_rate_32bit",
                "reflectivity_32bit",
                "mor_visibility",
                "weather_code_synop_4680",
                "weather_code_synop_4677",
                "raw_drop_number",
            ]
            df.columns = column_names
            # - Add time column
            df["time"] = df_time

        else:
            # Wrong number of delimiters ... likely a corrupted file
            raise SyntaxError("Something wrong with columns number!")

        # - Detect corrupted row by analyzing raw_drop_number
        # TODO: to be discarded in future by timestep ...
        df = df[df["raw_drop_number"].str.contains("0p0") == False]

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_id>
    files_glob_pattern = "*.tar"

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
