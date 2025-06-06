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
    column_names = [
        "date",
        "time",
        "sensor_battery_voltage",
        "laser_amplitude",
        "sensor_heating_current",
        "sensor_temperature",
        "number_particles",
        "mor_visibility",
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        "reflectivity_32bit",
        "weather_code_nws",
        "weather_code_metar_4678",
        "raw_drop_number",
    ]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ","

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"

    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"

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

    # Skip first row as columns names
    reader_kwargs["header"] = None

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
    # Define datetime 'time' column
    df["time"] = df["date"] + "-" + df["time"]
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d-%H:%M:%S", errors="coerce")

    # Preprocess the raw spectrum
    # - The '<SPECTRUM>ZERO</SPECTRUM>' indicates no drops detected
    # --> "" generates an array of zeros in L0B processing
    df["raw_drop_number"] = df["raw_drop_number"].astype("string")
    df["raw_drop_number"] = df["raw_drop_number"].str.strip()
    df["raw_drop_number"] = df["raw_drop_number"].str.replace("<SPECTRUM>ZERO</SPECTRUM>", "")

    # Remove <SPECTRUM> and </SPECTRUM>" acronyms from the raw_drop_number field
    df["raw_drop_number"] = df["raw_drop_number"].str.replace("<SPECTRUM>", "")
    df["raw_drop_number"] = df["raw_drop_number"].str.replace("</SPECTRUM>", "")

    # Preprocess the raw spectrum and raw_drop_average_velocity
    # - Add 0 before every ; if ; not preceded by a digit
    # - Example: ';;1;;' --> '0;0;1;0;'
    df["raw_drop_number"] = df["raw_drop_number"].str.replace(r"(?<!\d);", "0;", regex=True)

    # Drop columns not agreeing with DISDRODB L0 standards
    df = df.drop(columns=["date"])

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
