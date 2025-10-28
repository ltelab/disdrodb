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
"""DISDRODB reader for University of Bergen OTT Parsivel 2 raw data."""

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

    # - Skip first row as columns names
    reader_kwargs["header"] = None

    # - Skip header
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
    # reader_kwargs['compression'] = 'xz'

    # - Strings to recognize as NA/NaN and replace with standard NA flags
    #   - Already included: '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
    #                       '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
    #                       'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
    reader_kwargs["na_values"] = ["na", "error", "-.-", " NA"]

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
    # Remove corrupted rows
    df = df[df["TO_PARSE"].str.count(";") == 1101]

    # Split into columns
    df = df["TO_PARSE"].str.split(";", expand=True, n=13)

    # Assign columns names
    names = [
        "date",
        "time",
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "snowfall_rate",
        "weather_code_synop_4680",
        "reflectivity_32bit",
        "mor_visibility",
        "rain_kinetic_energy",
        "sensor_temperature",
        "laser_amplitude",
        "number_particles",
        "sensor_battery_voltage",
        "TO_SPLIT",
    ]
    df.columns = names

    # Sanitize date
    date = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    date = date.ffill().bfill()

    # Add datetime time column
    time_str = date.astype(str) + "T" + df["time"]
    df["time"] = pd.to_datetime(time_str, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    df = df.drop(columns=["date"])

    # Derive raw drop arrays
    df_split = df["TO_SPLIT"].str.split(";", expand=True)
    df["raw_drop_concentration"] = df_split.iloc[:, :32].agg(";".join, axis=1)
    df["raw_drop_average_velocity"] = df_split.iloc[:, 32:64].agg(";".join, axis=1)
    df["raw_drop_number"] = df_split.iloc[:, 64:].agg(";".join, axis=1)
    del df_split

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "TO_SPLIT",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
