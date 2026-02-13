#!/usr/bin/env python3
# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
"""This reader allows to read raw data from NASA KWAJALEIN stations."""

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
    column_names = ["TO_PARSE"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "//n"
    # - Skip first row as columns names
    reader_kwargs["header"] = None
    reader_kwargs["skiprows"] = 0
    # - Define encoding
    reader_kwargs["encoding"] = "latin-1"
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
    #   - Already included: '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
    #                       '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
    #                       'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
    reader_kwargs["na_values"] = ["na", "", "error", "NA", "-.-"]

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
    # Remove rows with invalid number of separators
    df = df[df["TO_PARSE"].str.count(" ") == 1030]
    if len(df) == 0:
        raise ValueError(f"No valid data in {filepath}")

    # Retrieve time and telegram field
    df = df["TO_PARSE"].str.split(" ", expand=True, n=6)
    df.columns = ["year", "month", "day", "dummy", "hour", "minute", "TO_BE_JOINED"]

    # Create datetime rounded to minute (seconds default to 00)
    df["time"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute"]],
    )
    # Define seconds (consecutive 5 rows with same minute --> infer 10 seconds increments)
    df["second"] = df.groupby("time").cumcount() * 10
    df["time"] = df["time"] + pd.to_timedelta(df["second"], unit="s")

    # Join raw drop number
    df["raw_drop_number"] = df["TO_BE_JOINED"].str.replace(" ", ",").str.strip(",")

    # Drop columns not agreeing with DISDRODB L0 standards
    df = df.drop(columns=["year", "month", "day", "hour", "minute", "second", "dummy", "TO_BE_JOINED"])

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
