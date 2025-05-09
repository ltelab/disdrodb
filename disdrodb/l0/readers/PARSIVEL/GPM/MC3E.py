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
    column_names = ["time", "TO_BE_SPLITTED"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ";"
    # - Skip first row as columns names
    reader_kwargs["header"] = None
    # - Skip file with encoding errors
    reader_kwargs["encoding_errors"] = "ignore"
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
    reader_kwargs["na_values"] = ["na", "", "error", "-.-"]

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
    # Convert 'time' column to datetime
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S", errors="coerce")

    # Count number of delimiters in the column to be parsed
    # --> Some first rows are corrupted, so count the most frequent occurrence
    possible_delimiters, counts = np.unique(df["TO_BE_SPLITTED"].str.count(","), return_counts=True)
    n_delimiters = possible_delimiters[np.argmax(counts)]

    # ---------------------------------------------------------
    #### Case of 1031 delimiters
    if n_delimiters == 1031:  # first files
        # Select valid rows
        df = df.loc[df["TO_BE_SPLITTED"].str.count(",") == 1031]
        # Get time column
        df_time = df["time"]
        # Split the 'TO_BE_SPLITTED' column
        df = df["TO_BE_SPLITTED"].str.split(",", expand=True, n=7)
        # Assign column names
        column_names = [
            "station_name",
            "sensor_status",
            "sensor_temperature",
            "reflectivity_32bit",
            "mor_visibility",
            "weather_code_synop_4680",
            "weather_code_synop_4677",
            "raw_drop_number",
        ]
        df.columns = column_names
        # Add time column
        df["time"] = df_time
        # Remove columns not in other files
        df = df.drop(columns="reflectivity_32bit")
        # Add missing columns and set NaN value
        missing_columns = [
            "number_particles",
            "rainfall_rate_32bit",
            "reflectivity_16bit",
        ]
        for column in missing_columns:
            df[column] = "NaN"
        # Drop columns not agreeing with DISDRODB L0 standards
        df = df.drop(columns=["station_name"])
        # Return the dataframe adhering to DISDRODB L0 standards
        return df
    # ---------------------------------------------------------
    #### Case of 1032 delimiters
    if n_delimiters == 1033:  # (most of the files)
        # Select valid rows
        df = df.loc[df["TO_BE_SPLITTED"].str.count(",") == 1033]
        # Get time column
        df_time = df["time"]
        # Split the column be parsed
        df = df["TO_BE_SPLITTED"].str.split(",", expand=True, n=9)
        # Assign column names
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
        # Add time column
        df["time"] = df_time
        # Drop columns not agreeing with DISDRODB L0 standards
        df = df.drop(columns=["station_name"])
        # Return the dataframe adhering to DISDRODB L0 standards
        return df

    # ---------------------------------------------------------
    #### Case of 1035 delimiters
    if n_delimiters == 1035:  # APU 17 first files
        # Select valid rows
        df = df.loc[df["TO_BE_SPLITTED"].str.count(",") == 1035]
        # Get time column
        df_time = df["time"]
        # Split the column be parsed
        df = df["TO_BE_SPLITTED"].str.split(",", expand=True, n=11)
        # Assign column names
        column_names = [
            "station_name",
            "sensor_date",
            "sensor_time",
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
        # Add time column
        df["time"] = df_time
        # Drop columns not needed
        df = df.drop(columns=["sensor_time", "sensor_date"])
        # Drop columns not agreeing with DISDRODB L0 standards
        df = df.drop(columns=["station_name"])
        # Return the dataframe adhering to DISDRODB L0 standards
        return df

    # ---------------------------------------------------------
    #### Undefined number of delimiters
    # - Likely a corrupted file
    raise ValueError(f"Unexpected number of comma delimiters: {n_delimiters} !")
