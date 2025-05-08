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
"""Reader for CHONGQING campaign."""
import numpy as np
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
    column_names = ["TO_SPLIT"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    reader_kwargs["delimiter"] = ","
    # - No header
    # reader_kwargs["header"] = None
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
    #   - Already included: '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
    #                       '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
    #                       'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
    reader_kwargs["na_values"] = ["na", "", "error", "-.-", "C" * 32]

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
    # Drop invalid rows
    # --> C*32 that is nan
    df = df.dropna()

    # Check if there are valid data
    if len(df) == 0:
        raise ValueError("No data available")

    # Retrieve timesteps
    df_time = df["TO_SPLIT"].iloc[0::33]
    df_time = pd.to_datetime(df_time, format="%Y.%m.%d;%H:%M", errors="coerce")
    df_time = df_time.rename("time")
    df_time = df_time.reset_index(drop=True)

    # Retrieve data
    idx = np.ones(len(df)).astype(bool)
    idx[0::33] = False
    df_data = df[idx]

    # Check consistency (no missing rows)
    n_expected_data_rows = int(len(df_time) * 32)
    if len(df_data) != n_expected_data_rows:
        raise ValueError("Not same amount of timesteps and data.")

    # Replace heterogeneous number of spaces with a single space
    df_data["TO_SPLIT"] = df_data["TO_SPLIT"].str.replace(r" +", " ", regex=True).str.strip(" ")

    # Retrieve arrays
    arr = df_data["TO_SPLIT"].str.split(" ", expand=True).to_numpy()
    # Flatten by row and then reshape to n_timesteps x 1024
    arr = arr.flatten(order="C").reshape(len(df_time), 1024)
    # Then concat all the 1024 bins
    df_arr = pd.DataFrame(arr, dtype="str")
    df_raw_drop_number = df_arr.agg(",".join, axis=1)
    df_raw_drop_number = df_raw_drop_number.rename("raw_drop_number")

    # Create dataframe
    df = pd.concat([df_time, df_raw_drop_number], axis=1)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
