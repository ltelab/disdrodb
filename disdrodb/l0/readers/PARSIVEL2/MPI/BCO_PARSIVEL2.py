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
    column_names = ["TO_BE_PARSED"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = "/\n"

    # - Skip first row as columns names
    reader_kwargs["header"] = None

    # - Skip first 2 rows
    reader_kwargs["skiprows"] = 0

    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"

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
    reader_kwargs["na_values"] = ["na", "", "error", "NA"]

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
    # Raise error if empty file
    if len(df) == 0:
        raise ValueError(f"{filepath} is empty.")

    # Select only rows with expected number of delimiters
    df = df[df["TO_BE_PARSED"].str.count(";") == 1107]

    # Raise error if no data left
    if len(df) == 0:
        raise ValueError(f"No valid data in {filepath}.")

    # Split by ; delimiter
    df = df["TO_BE_PARSED"].str.split(";", expand=True, n=19)

    # Assign column names
    names = [
        "date",
        "time",
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        # "weather_code_synop_4677",
        # "weather_code_metar_4678",
        "reflectivity_32bit",
        "mor_visibility",
        "sample_interval",
        "laser_amplitude",
        "number_particles",
        "sensor_temperature",
        "sensor_serial_number",
        "firmware_iop",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "sensor_status",
        "station_name",
        "rainfall_amount_absolute_32bit",
        "error_code",
        "ARRAY_TO_SPLIT",
    ]

    df.columns = names

    # Define time in datetime format
    time_str = df["date"] + " " + df["time"]
    df["time"] = pd.to_datetime(time_str, format="%d.%m.%Y %H:%M:%S", errors="coerce")

    # Add raw array
    df["raw_drop_concentration"] = df["ARRAY_TO_SPLIT"].str[:224]
    df["raw_drop_average_velocity"] = df["ARRAY_TO_SPLIT"].str[224:448]
    df["raw_drop_number"] = df["ARRAY_TO_SPLIT"].str[448:]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "date",
        "station_name",
        "firmware_iop",
        "ARRAY_TO_SPLIT",
        "sensor_serial_number",
        "sample_interval",
    ]
    df = df.drop(columns=columns_to_drop)
    return df
