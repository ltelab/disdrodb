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
    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    def df_sanitizer_fun(df, lazy=lazy):

        # - Import dask or pandas
        if lazy:
            import dask.dataframe as dd
        else:
            import pandas as dd

        temp_time = df.loc[df.iloc[:, 0].astype(str).str.len() == 16].add_prefix("col_")
        temp_time["col_0"] = dd.to_datetime(temp_time["col_0"], format="%Y.%m.%d;%H:%M")

        # Insert Raw into a series and drop last line
        temp_raw = df.loc[df.iloc[:, 0].astype(str).str.len() != 16].add_prefix("col_")
        temp_raw["col_0"] = temp_raw["col_0"].str.lstrip("   ")
        temp_raw = temp_raw.dropna()

        # If raw_drop_number series is not a 32 multiple, throw error
        if len(temp_raw) % 32 != 0:
            msg = "Wrong column number on raw_drop_number, can not parse!"
            raise ValueError(msg)

        # Series and variable temporary for parsing raw_drop_number
        if lazy:
            import pandas as pd

            raw = pd.DataFrame({"raw_drop_number": []})
            # raw = dd.from_pandas(raw, npartitions=1, chunksize=None)
        else:
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

        if lazy:
            raw = dd.from_pandas(raw, npartitions=1, chunksize=None)

        # Reset all index
        temp_time = temp_time.reset_index(drop=True)
        raw = raw.reset_index(drop=True)

        if lazy:
            df = dd.concat([temp_time, raw], axis=1, ignore_unknown_divisions=True)
        else:
            df = dd.concat([temp_time, raw], axis=1)

        df.columns = ["time", "raw_drop_number"]

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
