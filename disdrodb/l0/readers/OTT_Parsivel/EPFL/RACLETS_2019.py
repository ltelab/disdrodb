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
"""Reader for RACLETS 2019 campaign."""
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
    column_names = ["TO_BE_SPLITTED"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ";"  # Used to not split anything !
    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"
    # Skip first row as columns names
    reader_kwargs["header"] = None
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
    reader_kwargs["compression"] = "gzip"
    # - Strings to recognize as NA/NaN and replace with standard NA flags
    #   - Already included: '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
    #                       '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
    #                       'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
    reader_kwargs["na_values"] = [
        "na",
        "",
        "error",
    ]

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
    # Drop row that contains errors
    df = df[~df["TO_BE_SPLITTED"].str.contains("Error in data reading! 0")]

    # Check if file empty
    if len(df.index) == 0:
        raise ValueError("Error in all rows. The file has been skipped.")

    # Split the column
    df = df["TO_BE_SPLITTED"].str.split(",", expand=True, n=1111)

    # Define auxiliary columns
    column_names = [
        "id",
        "latitude",
        "longitude",
        "time",
        "datalogger_temperature",
        "datalogger_voltage",
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
        "rainfall_amount_absolute_32bit",
        "error_code",
    ]
    df_variables = df.iloc[:, 0:20]
    df_variables.columns = column_names

    # Define raw fields
    df_raw_drop_concentration = df.iloc[:, 20:52].agg(",".join, axis=1)
    df_raw_drop_average_velocity = df.iloc[:, 53:85].agg(",".join, axis=1)
    df_raw_drop_number = df.iloc[:, 86:1110].agg(",".join, axis=1)

    # Combine together
    df = df_variables
    df["raw_drop_concentration"] = df_raw_drop_concentration
    df["raw_drop_average_velocity"] = df_raw_drop_average_velocity
    df["raw_drop_number"] = df_raw_drop_number

    # Drop invalid rows
    df = df.loc[df["id"].astype(str).str.len() < 10]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "datalogger_temperature",
        "datalogger_voltage",
        "id",
        "latitude",
        "longitude",
    ]
    df = df.drop(columns=columns_to_drop)

    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
