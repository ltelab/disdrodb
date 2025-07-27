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
    column_names = ["TO_PARSE"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
    # - Skip first row as columns names
    # - Define encoding
    reader_kwargs["encoding"] = "latin"  # "ISO-8859-1"
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
    # Identify groups of lines corresponding to same measurement
    is_start = df["TO_PARSE"].str.startswith("#0#,")
    df["observation_id"] = is_start.cumsum()

    # Loop over groups and create a dataframe with a single row for each measurement
    list_obs = []
    for _, df_obs in df.groupby("observation_id", sort=False):
        if len(df_obs) not in [6, 7]:
            pass

        # Remove #<id># and last comma
        series = df_obs["TO_PARSE"].str.split(",", n=1, expand=True)[1].str.rstrip(",")
        if len(df_obs) == 7:
            series = series.iloc[0:6]

        # Create dataframe and name columns
        df_obs = series.to_frame().T
        df_obs.columns = [
            "time",
            "TO_SPLIT1",
            "TO_SPLIT2",
            "raw_drop_concentration",
            "raw_drop_average_velocity",
            "raw_drop_number",
        ]

        # Append to the list
        list_obs.append(df_obs)

    # Concat all timesteps into a single dataframe
    df = pd.concat(list_obs)

    # Split and rename remaining variables
    df_split1 = df["TO_SPLIT1"].str.split(",", expand=True)
    df_split1.columns = ["weather_code_synop_4680", "unknown1", "unknown2", "reflectivity_32bit"]
    df_split2 = df["TO_SPLIT2"].str.split(",", expand=True)
    df_split2.columns = ["parsivel_id", "unknown3", "mor_visibility", "laser_amplitude", "sensor_status"]

    # Merge everything into a single dataframe
    df = pd.concat([df, df_split1, df_split2], axis=1)

    # Define time as datetime64
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")

    # Remove unused variables
    df = df.drop(columns=["TO_SPLIT1", "TO_SPLIT2", "parsivel_id", "unknown1", "unknown2", "unknown3"])

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
