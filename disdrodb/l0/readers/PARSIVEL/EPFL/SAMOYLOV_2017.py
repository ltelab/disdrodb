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
"""Reader for SAMOYLOV 2017 campaign."""
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
    column_names = [
        "time",
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
        "raw_drop_concentration",
        "raw_drop_average_velocity",
        "raw_drop_number",
    ]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = ";"

    # Skip first row as columns names
    reader_kwargs["header"] = None

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"

    # - Avoid UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf0
    reader_kwargs["encoding"] = "latin-1"

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

    # Different encoding for this campaign
    reader_kwargs["encoding"] = "latin-1"

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

    # Special parsing if 'Error in data reading' in rainfall_rate_32bit column
    if np.any(df["rainfall_rate_32bit"].str.startswith("Error in data reading!", na=False)):
        df["rainfall_rate_32bit"] = df["rainfall_rate_32bit"].str.replace("Error in data reading!", "")
        df["raw_drop_number"] = df["raw_drop_average_velocity"]
        df["raw_drop_average_velocity"] = df["raw_drop_concentration"]
        df["raw_drop_concentration"] = df["error_code"]
        df["error_code"] = df["rainfall_amount_absolute_32bit"].str[7:]
        df["rainfall_amount_absolute_32bit"] = df["rainfall_amount_absolute_32bit"].str[:7]

    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
