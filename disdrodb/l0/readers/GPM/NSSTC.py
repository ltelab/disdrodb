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
from disdrodb.l0 import run_l0a
from disdrodb.l0.l0_reader import reader_generic_docstring, is_documented_by


@is_documented_by(reader_generic_docstring)
def reader(
    raw_dir,
    processed_dir,
    station_name,
    # Processing options
    force=False,
    verbose=False,
    parallel=False,
    debugging_mode=False,
):
    ##------------------------------------------------------------------------.
    #### - Define column names
    column_names = ["time", "TO_BE_SPLITTED"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
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

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    # Deal with changing file format after 25 feb 2011 by the documentation
    # - https://ghrc.nsstc.nasa.gov/pub/fieldCampaigns/gpmValidation/relatedProjects/nsstc/parsivel/doc/gpm_parsivel_nsstc_dataset.html).

    def df_sanitizer_fun(df):
        # - Import pandas
        import numpy as np
        import pandas as pd

        # - Check 'time' string length to detect corrupted rows
        df = df[df["time"].str.len() == 14]

        # - Convert time column to datetime
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S", errors="coerce")

        # Count number of delimiters in the column to be parsed
        # --> Some first rows are corrupted, so count the most frequent occurence
        possible_delimeters, counts = np.unique(df["TO_BE_SPLITTED"].str.count(","), return_counts=True)
        n_delimiters = possible_delimeters[np.argmax(counts)]

        if n_delimiters == 1027:
            # - Select valid rows
            df = df.loc[df["TO_BE_SPLITTED"].str.count(",") == 1027]
            # - Get time column
            df_time = df["time"]
            # - Split the 'TO_BE_SPLITTED' column
            df = df["TO_BE_SPLITTED"].str.split(",", expand=True, n=3)
            # - Assign column names
            column_names = [
                "station_name",
                "sensor_status",
                "sensor_temperature",
                "raw_drop_number",
            ]
            df.columns = column_names
            # - Add time column
            df["time"] = df_time
            # - Add missing columns and set NaN value
            missing_columns = [
                "number_particles",
                "rainfall_rate_32bit",
                "reflectivity_16bit",
                "mor_visibility",
                "weather_code_synop_4680",
                "weather_code_synop_4677",
            ]
            for column in missing_columns:
                df[column] = "NaN"
        elif n_delimiters == 1033:
            # - Select valid rows
            df = df.loc[df["TO_BE_SPLITTED"].str.count(",") == 1033]
            # - Get time column
            df_time = df["time"]
            # - Split the column be parsed
            df = df["TO_BE_SPLITTED"].str.split(",", expand=True, n=9)
            # - Assign column names
            column_names = [
                "station_name",
                "sensor_status",
                "sensor_temperature",
                "number_particles",
                "rainfall_rate_32bit",
                "reflectivity_16bit",
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
            raise ValueError("Unexpected number of comma delimiters !")

        # - Drop columns not agreeing with DISDRODB L0 standards
        df = df.drop(columns=["station_name"])

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_name>
    glob_patterns = "*.dat"

    ####----------------------------------------------------------------------.
    #### - Create L0A products
    run_l0a(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        station_name=station_name,
        # Custom arguments of the reader for L0A processing
        glob_patterns=glob_patterns,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
        # Processing options
        force=force,
        verbose=verbose,
        parallel=parallel,
        debugging_mode=debugging_mode,
    )
