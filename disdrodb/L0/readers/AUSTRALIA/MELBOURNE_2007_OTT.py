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
    column_names = [
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        "weather_code_synop_4677",
        "weather_code_metar_4678",
        "reflectivity_32bit",
        "mor_visibility",
        "laser_amplitude",
        "number_particles",
        "unknow2",
        "datalogger_temperature",
        "sensor_status",
        "station_name",
        "unknow3",
        "unknow4",
        "error_code",
        "TO_BE_SPLITTED",
    ]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = "!"

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
    reader_kwargs["na_values"] = ["na", "", "error", "NA", "NP   "]

    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    # Cast all to string
    reader_kwargs["dtype"] = str

    # Skip first row as columns names
    reader_kwargs["header"] = None

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer
    # - Enable to deal with bad raw data files
    # - Enable to standardize raw data files to L0 standards  (i.e. time to datetime)

    def df_sanitizer_fun(df, lazy=False):
        # Import dask or pandas
        if lazy:
            import dask.dataframe as dd
        else:
            import pandas as dd

        # Data format:
        # -2015-01-09 00:02:16
        # 0000.063;0012.33;51;51;  -DZ; ...

        # Convert 'temp' column to string
        df["temp"] = df["temp"].astype(str)

        # Infer time
        df_time = df.loc[df["temp"].str.len() == 20]
        df_time["time"] = dd.to_datetime(df_time["temp"], format="-%Y-%m-%d %H:%M:%S")
        df_time = df_time.drop(columns=["temp"])

        # Drop header's log and corrupted rows
        df = df.loc[df["temp"].str.len() > 620]

        # Split first 80 columns
        df = df["temp"].str.split(";", n=16, expand=True)
        df.columns = column_names

        # Retrieve raw_drop* fields
        df["raw_drop_concentration"] = df["TO_BE_SPLITTED"].str[:224]
        df["raw_drop_average_velocity"] = df["TO_BE_SPLITTED"].str[224:448]
        df["raw_drop_number"] = df["TO_BE_SPLITTED"].str[448:]

        # Concat df and df_time
        df = df.reset_index(drop=True)
        df_time = df_time.reset_index(drop=True)
        df = dd.concat([df_time, df], axis=1, ignore_unknown_divisions=True)

        # Drop last columns (all nan)
        df = df.dropna(thresh=(len(df.columns) - 19), how="all")

        # Drops columns not compliant with DISDRODB L0 standard
        columns_to_drop = [
            "TO_BE_SPLITTED",
            "datalogger_temperature",
            "station_name",
            "unknow2",
            "unknow3",
            "unknow4",
        ]
        df = df.drop(columns=columns_to_drop)

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_id>
    files_glob_pattern = "*.txt*"

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
