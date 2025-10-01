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
"""Reader for the OTT Parsivel2 sensors of the CW3E network."""
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
    # Skip first row as columns names
    reader_kwargs["header"] = None
    # Skip file with encoding errors
    reader_kwargs["encoding_errors"] = "ignore"
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
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
    df = df[df["TO_PARSE"].str.count(";") == 1105]

    # Split the columns
    df = df["TO_PARSE"].str.split(";", n=16, expand=True)

    # Assign column names
    names = [
        "sensor_serial_number",
        "sensor_status",
        "laser_amplitude",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "dummy_date",
        "sensor_time",
        "sensor_date",
        "sensor_temperature",
        "number_particles",
        "rainfall_rate_32bit",
        "reflectivity_32bit",
        "rainfall_accumulated_16bit",
        "mor_visibility",
        "weather_code_synop_4680",
        "weather_code_synop_4677",
        "TO_SPLIT",
    ]
    df.columns = names

    # Derive raw arrays
    df_split = df["TO_SPLIT"].str.split(";", expand=True)
    df["raw_drop_concentration"] = df_split.iloc[:, :32].agg(",".join, axis=1)
    df["raw_drop_average_velocity"] = df_split.iloc[:, 32:64].agg(",".join, axis=1)
    df["raw_drop_number"] = df_split.iloc[:, 64:1088].agg(",".join, axis=1)
    df["rain_kinetic_energy"] = df_split.iloc[:, 1088]
    df["CHECK_EMPTY"] = df_split.iloc[:, 1089]

    # Ensure valid observation
    df = df[df["CHECK_EMPTY"] == ""]

    # Add the time column
    time_str = df["sensor_date"] + "-" + df["sensor_time"]
    df["time"] = pd.to_datetime(time_str, format="%d.%m.%Y-%H:%M:%S", errors="coerce")

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "dummy_date",
        "sensor_date",
        "sensor_time",
        "sensor_serial_number",
        "rainfall_accumulated_16bit",  # unexpected format
        "CHECK_EMPTY",
        "TO_SPLIT",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
