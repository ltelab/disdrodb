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
"""Reader for UNIL 2022 campaign."""
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
        "id",
        "latitude",
        "longitude",
        "time",
        "datalogger_temperature",
        "TO_BE_SPLITTED",  # datalogger_voltage and rainfall_rate_32bit
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
        "raw_drop_concentration",
        "raw_drop_average_velocity",
        "raw_drop_number",
        "datalogger_error",
    ]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ";"
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
    # Convert time column to datetime with resolution in seconds
    df["time"] = pd.to_datetime(df["time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")

    # Split TO_BE_SPLITTED columns
    df_splitted = df["TO_BE_SPLITTED"].str.split(",", expand=True, n=1)
    df_splitted.columns = ["datalogger_voltage", "rainfall_rate_32bit"]
    df["rainfall_rate_32bit"] = df_splitted["rainfall_rate_32bit"]

    # Remove rows with error in data reading
    # - When datalogger error: rainfall_rate_32bit: Error in data reading!
    df = df[df["rainfall_rate_32bit"] != "Error in data reading! 0"]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "id",
        "TO_BE_SPLITTED",
        "datalogger_temperature",
        "datalogger_error",
        "latitude",
        "longitude",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
