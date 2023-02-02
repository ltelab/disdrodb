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
"""Reader for CHONGQING campaign."""
from disdrodb.L0 import run_l0a
from disdrodb.L0.L0_reader import reader_generic_docstring, is_documented_by


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
    column_names = []

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - No header
    reader_kwargs["header"] = None
    # - Define encoding
    reader_kwargs["encoding"] = "latin-1"
    # - Avoid first column to become df index
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
    reader_kwargs["na_values"] = ["na", "", "error", "-.-", ["C"] * 32]

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    def df_sanitizer_fun(df):
        # - Import pandas
        import pandas as pd

        temp_time = df.loc[df.iloc[:, 0].astype(str).str.len() == 16].add_prefix("col_")
        temp_time["col_0"] = pd.to_datetime(temp_time["col_0"], format="%Y.%m.%d;%H:%M")

        # Insert Raw into a series and drop last line
        temp_raw = df.loc[df.iloc[:, 0].astype(str).str.len() != 16].add_prefix("col_")
        temp_raw["col_0"] = temp_raw["col_0"].str.lstrip("   ")
        temp_raw = temp_raw.dropna()

        # If raw_drop_number series is not a 32 multiple, throw error
        if len(temp_raw) % 32 != 0:
            msg = "Wrong column number on raw_drop_number, can not parse!"
            raise ValueError(msg)

        # Series and variable temporary for parsing raw_drop_number
        raw = pd.DataFrame({"raw_drop_number": []})

        temp_string_2 = ""

        # Parse for raw_drop_number
        for index, value in temp_raw.iterrows():
            temp_string = ""

            if index % 32 != 0:
                temp_string = value["col_0"].split(" ")

                # Remove blank string from split
                temp_string = list(filter(None, temp_string))

                # Cast list to int
                temp_string = [int(i) for i in temp_string]

                # Join list separate by comma
                temp_string = ", ".join(map(str, temp_string))

                # Add last comma
                temp_string += ", "

                temp_string_2 += temp_string

            else:
                raw = raw.append({"raw_drop_number": temp_string_2}, ignore_index=True)

                temp_string_2 = ""

        # Reset all index
        temp_time = temp_time.reset_index(drop=True)
        raw = raw.reset_index(drop=True)

        df = pd.concat([temp_time, raw], axis=1)
        df.columns = ["time", "raw_drop_number"]

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_name>
    glob_patterns = "*.txt*"

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
