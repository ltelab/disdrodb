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
import os

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file


def reader_parsivel(filepath, logger):
    """Reader for Parsivel CR1000 Data Logger file."""
    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ","
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
        column_names=None,
        reader_kwargs=reader_kwargs,
        logger=logger,
    )

    #### Define column names
    n_columns = len(df.columns)
    if n_columns == 16:
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
            "sensor_status",
            "rain_kinetic_energy",
            "V_Batt_Min",
        ]
    elif n_columns == 20:
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
            "sensor_status",
            "sensor_temperature_receiver",
            "sensor_temperature_trasmitter",
            "rain_kinetic_energy",
            "V_Batt_Min",
            "sample_interval",
            "Temps_present",
        ]
    else:
        raise ValueError(f"{filepath} has {n_columns} columns. Undefined reader.")

    ##------------------------------------------------------------------------.
    #### Assign column names
    df.columns = column_names

    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Define time
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Set missing columns as NaN
    potential_missing_columns = [
        "sensor_temperature_receiver",
        "sensor_temperature_trasmitter",
        "rain_kinetic_energy",
    ]
    for column in potential_missing_columns:
        if column not in df.columns:
            df[column] = np.nan

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = ["RECORD", "V_Batt_Min", "Temps_present", "sample_interval"]
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
    # Define time
    df = df["TO_PARSE"].str.split(",", n=2, expand=True)
    df.columns = ["time", "RECORD", "TO_PARSE"]

    df["time"] = pd.to_datetime(df["time"].str.strip('"'), format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Derive raw drop arrays
    def split_string(s):
        vals = [v.strip() for v in s.split(",")]
        c1 = ",".join(vals[:32])  # -10
        c1 = c1.replace("-10", "0")
        c2 = "0,0," + ",".join(vals[32:62])
        c3 = ",".join(vals[62:])
        series = pd.Series(
            {
                "raw_drop_concentration": c1,
                "raw_drop_average_velocity": c2,
                "raw_drop_number": c3,
            },
        )
        return series

    splitted_string = df["TO_PARSE"].apply(split_string)
    df["raw_drop_concentration"] = splitted_string["raw_drop_concentration"]
    df["raw_drop_average_velocity"] = splitted_string["raw_drop_average_velocity"]
    df["raw_drop_number"] = splitted_string["raw_drop_number"]

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
    df["raw_drop_concentration"] = ""
    df["raw_drop_average_velocity"] = ""
    df["raw_drop_number"] = ""

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
