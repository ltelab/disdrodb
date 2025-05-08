#!/usr/bin/env python3

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2023 DISDRODB developers
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
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = [
        "date",
        "time",
        "sensor_status",
        "sample_interval",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "n10",
        "n11",
        "n12",
        "n13",
        "n14",
        "n15",
        "n16",
        "n17",
        "n18",
        "n19",
        "n20",
        "RI",
        "RA",
        "RAT",
    ]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\t"
    # Skip header
    reader_kwargs["header"] = None
    # Skip first row as columns names
    reader_kwargs["skiprows"] = 1
    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"
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
    #   - Already included: '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
    #                       '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
    #                       'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
    reader_kwargs["na_values"] = ["na", "", "error"]

    ##------------------------------------------------------------------------.
    #### Read the data
    df = read_raw_text_file(
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        logger=logger,
    )

    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Replace 'status' NaN with 0
    df["sensor_status"] = df["sensor_status"].astype(float).fillna(value=0).astype(int)

    # Define 'time' datetime column
    df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.drop(columns=["date"])

    # Create raw_drop_number column
    bin_columns = ["n" + str(i) for i in range(1, 21)]
    df_arr = df[bin_columns]
    df_raw_drop_number = df_arr.agg(";".join, axis=1)
    df["raw_drop_number"] = df_raw_drop_number

    # Remove bins columns
    df = df.drop(columns=bin_columns)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
