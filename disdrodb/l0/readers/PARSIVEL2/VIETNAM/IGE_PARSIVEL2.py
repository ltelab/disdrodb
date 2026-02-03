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
import os

import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file


def reader_parsivel(filepath, logger):
    """Reader for Parsivel CR1000 Data Logger file."""
    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
    # - Skip first row as columns names
    reader_kwargs["header"] = None
    # - Skip first 3 rows
    reader_kwargs["skiprows"] = 0
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
    df_raw = read_raw_text_file(
        filepath=filepath,
        column_names=["TO_PARSE"],
        reader_kwargs=reader_kwargs,
        logger=logger,
    )

    # Retrieve header, number of columns and starting rows
    # - Search in the first 3 rows where "TIMESTAMP" occurs
    # - Once identified the row, strip away everything before TIMESTAMP
    # - Then identify start_row_idx as the row where "TIMESTAMP" occurs + 2
    for i in range(3):
        line = df_raw.iloc[i]["TO_PARSE"]
        if "TIMESTAMP" in line:
            # Remove double and single quotes
            line = line.replace('""', '"').replace('"', "")
            # Define header
            timestamp_idx = line.find("TIMESTAMP")
            header_str = line[timestamp_idx:]
            header = header_str.split(",")
            # Define number of columns
            n_columns = len(header)
            # Define start row with data
            start_row_idx = i + 3
            break
    else:
        # start_row_idx = 0
        # n_columns = len(df_raw["TO_PARSE"].iloc[0].split(","))
        raise ValueError("Could not find 'TIMESTAMP' in the first 3 rows of the file.")

    # Retrieve rows with actual data
    df = df_raw.iloc[start_row_idx:]

    # Expand dataframe
    df = df["TO_PARSE"].str.split(",", expand=True, n=n_columns - 1)

    #### Define column names
    column_names = [
        "time",
        "RECORD",
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        "weather_code_synop_4677",
        "reflectivity_32bit",
        "mor_visibility",
        "laser_amplitude",
        "number_particles",
        "sensor_temperature",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "sample_interval",
        "sensor_status",
        "rain_kinetic_energy",
        "sensor_temperature_receiver",
        "sensor_temperature_trasmitter",
        "V_Batt_Min",
    ]

    ##------------------------------------------------------------------------.
    #### Assign column names
    df.columns = column_names

    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Define time as datetime column
    df["time"] = pd.to_datetime(df["time"].str.strip('"'), format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "RECORD",
        "V_Batt_Min",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df


def reader_spectrum(filepath, logger):
    """Reader for Spectrum CR1000 Data Logger file."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = ["TO_PARSE"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
    # - Skip first row as columns names
    reader_kwargs["header"] = None
    # - Skip first 3 rows
    reader_kwargs["skiprows"] = 4
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
    # Split and assign columns
    df = df["TO_PARSE"].str.split(",", n=2, expand=True)
    df.columns = ["time", "RECORD", "TO_PARSE"]

    # Define time in datetime format
    df["time"] = pd.to_datetime(df["time"].str.strip('"'), format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Keep only rows with valid number of values
    df = df[df["TO_PARSE"].str.count(",") == 1085]

    # Retrieve arrays
    df_split = df["TO_PARSE"].str.split(",", expand=True)
    raw_drop_concentration = df_split.iloc[:, :32].agg(",".join, axis=1).str.replace("-10", "0")
    raw_drop_average_velocity = "0,0," + df_split.iloc[:, 32:62].agg(",".join, axis=1)
    raw_drop_number = df_split.iloc[:, 62:].agg(",".join, axis=1)
    df["raw_drop_concentration"] = raw_drop_concentration
    df["raw_drop_average_velocity"] = raw_drop_average_velocity
    df["raw_drop_number"] = raw_drop_number

    # Drop columns not agreeing with DISDRODB L0 standards
    df = df.drop(columns=["TO_PARSE", "RECORD"])
    return df


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    # Retrieve spectrum filepath
    spectrum_filepath = filepath.replace("parsivel", "spectre")

    # Read integral variables
    df = reader_parsivel(filepath, logger=logger)

    # Drop duplicates timesteps
    df = df.drop_duplicates(subset="time", keep="first")

    # Initialize empty arrays
    # --> 0 values array produced in L0B
    arrays_columns = ["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number"]
    for c in arrays_columns:
        if c not in df:
            df[c] = ""

    # Add raw spectrum if available
    if os.path.exists(spectrum_filepath):
        # Read raw spectrum for corresponding timesteps
        df_raw_spectrum = reader_spectrum(spectrum_filepath, logger=logger)
        df_raw_spectrum = df_raw_spectrum.drop_duplicates(subset="time", keep="first")
        # Add raw array to df
        df = df.set_index("time")
        df_raw_spectrum = df_raw_spectrum.set_index("time")
        df.update(df_raw_spectrum)
        # Set back time as column
        df = df.reset_index()

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
