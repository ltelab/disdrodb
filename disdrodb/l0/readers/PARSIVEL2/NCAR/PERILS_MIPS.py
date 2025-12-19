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
"""Reader for the PERILS 2022 and 2023 MIPS and RADAPS files."""

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
    reader_kwargs["delimiter"] = "\\n"
    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"

    # Skip first row as columns names
    reader_kwargs["header"] = None

    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"

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
    # Split and assign integrated variables names
    df = df["TO_PARSE"].str.split(",", expand=True, n=6)
    names = [
        "year",
        "doy",
        "time",
        "sensor_temperature",
        "number_particles",
        "rainfall_rate_32bit",
        "TO_SPLIT",
    ]
    df.columns = names

    # Derive raw drop arrays
    # - raw_drop_number is missing !
    df_split = df["TO_SPLIT"].str.split(",", expand=True)
    df["raw_drop_concentration"] = df_split.iloc[:, :32].agg(",".join, axis=1)
    df["raw_drop_average_velocity"] = df_split.iloc[:, 32:64].agg(",".join, axis=1)

    # Define datetime time column
    df["year"] = df["year"].str.replace(".0", "")
    df["doy"] = df["doy"].str.replace(".0", "")
    df["time"] = df["time"].str.replace(".0", "")
    df["time"] = df["year"].astype(str) + "-" + df["doy"].astype(str) + " " + df["time"].astype(str)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%j %H%M%S", errors="coerce")

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "doy",
        "year",
        "TO_SPLIT",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
