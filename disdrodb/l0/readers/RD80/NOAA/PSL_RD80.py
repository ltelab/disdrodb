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
"""DISDRODB Reader for NOAA PSL RD80 stations."""
import os

import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file


def read_new_format(filepath, logger):
    """Read new format."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = [
        "time_interval",
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
        "Dmax",
        "RI",
        "RA",
        "Wg",
        "Z",
        "EF",
        "N0",
        "slope",
        "NumDrops",
        "SumRA",
    ]
    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = r"\s+"
    # Skip header
    reader_kwargs["header"] = None
    # Skip first row as columns names
    reader_kwargs["skiprows"] = 2
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

    # Replace -99.9900 values with NaN
    columns_to_replace = ["Dmax", "RI", "RA", "Wg", "Z", "EF", "N0", "slope"]
    df[columns_to_replace] = df[columns_to_replace].replace("-99.9900", "NaN")

    # Replace 'Z' -Inf with NaN
    df["Z"] = df["Z"].str.replace("-Inf", "NaN")

    # Deal with time interval column
    # - Split into start and end time
    df_time = df["time_interval"].str.split("-", expand=True)
    df_time.columns = ["start", "end"]

    # - Convert start/end MM:SS:SSS to timedelta
    def parse_time(t):
        minutes, seconds, milliseconds = map(int, t.split(":"))
        return pd.Timedelta(minutes=minutes, seconds=seconds, milliseconds=milliseconds)

    df_time["start"] = df_time["start"].apply(parse_time)
    df_time["end"] = df_time["end"].apply(parse_time)
    # - Wrap end time if it's less than start time (i.e., crosses 60:00 boundary)
    # --> 00:00 --> 60:00
    df_time.loc[df_time["end"] < df_time["start"], "end"] += pd.Timedelta(minutes=60)

    # Compute sample_interval in seconds as integer
    df["sample_interval"] = (df_time["end"] - df_time["start"]).dt.total_seconds().astype(int)

    # Define time
    # - Extract date-hour
    filename = os.path.basename(filepath)
    if filename.startswith("lab") or filename.startswith("bao0") or filename.startswith("mdt0"):
        date_hour_str = filename[4:11]
    else:
        date_hour_str = filename[3:10]
    date_hour = pd.to_datetime(date_hour_str, format="%y%j%H")
    df["time"] = date_hour + df_time["start"]

    # Create raw_drop_number column
    bin_columns = ["n" + str(i) for i in range(1, 21)]
    df_arr = df[bin_columns]
    df_raw_drop_number = df_arr.agg(";".join, axis=1)
    df["raw_drop_number"] = df_raw_drop_number

    # Remove bins columns
    df = df.drop(columns=bin_columns)

    # # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "time_interval",
        "NumDrops",
        "SumRA",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df


def read_old_format(filepath, logger):
    """Read old format."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = [
        "date",
        "time",
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
        "Dmax",
        "RI",
        "RA",
        "Wg",
        "Z",
        "EF",
        "N0",
        "slope",
    ]
    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = r"\s+"
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

    # Replace -99.9900 values with NaN
    columns_to_replace = ["Dmax", "RI", "RA", "Wg", "Z", "EF", "N0", "slope"]
    df[columns_to_replace] = df[columns_to_replace].replace("-99.9900", "NaN")

    # Replace 'Z' -Inf with NaN
    df["Z"] = df["Z"].str.replace("-Inf", "NaN")

    # Define 'time' datetime column
    df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
    df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
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


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    filename = os.path.basename(filepath)
    # station_name = filename[0:3]
    if filename[3] == "-":  # czc-050101-0052.txt
        return read_old_format(filepath, logger=logger)
    # czc2201220b20.txt
    return read_new_format(filepath, logger=logger)
