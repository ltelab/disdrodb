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
"""Reader for EPFL 2009 campaign."""
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

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"

    # Skip the first row (header)
    reader_kwargs["skiprows"] = 0

    # - Define encoding
    reader_kwargs["encoding"] = "latin"

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
    # Create ID and Value columns
    df = df["TO_PARSE"].str.split(";", expand=True, n=14)

    # Assign column names
    column_names = [
        "id",
        "time",
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        "reflectivity_32bit",
        "mor_visibility",
        "sensor_temperature",  # maybe
        "laser_amplitude",  # probably
        "number_particles",
        "sensor_status",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "error_code",
        "raw_drop_number",
    ]
    df.columns = column_names

    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H.%M.%S", errors="coerce")

    # Preprocess the raw spectrum
    # - Add 0 before every ; if ; not preceded by a digit
    # Example: ';;1;;' --> '0;0;1;0;'
    df["raw_drop_number"] = df["raw_drop_number"].str.replace("R;", "")
    df["raw_drop_number"] = df["raw_drop_number"].str.replace(r"(?<!\d);", "0;", regex=True)

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "id",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
