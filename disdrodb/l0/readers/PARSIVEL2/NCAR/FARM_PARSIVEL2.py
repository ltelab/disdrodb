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
"""Reader for CSWR FARM disdrometer data (used in PERILS and RELAMPAGO campaign)."""
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

    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"

    # Skip first row as columns names
    reader_kwargs["header"] = None
    reader_kwargs["skiprows"] = 2

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
    # Split and assign integrated variables names
    df = df["TO_PARSE"].str.split(",", expand=True, n=22)

    names = [
        "time",
        "station_name",
        "station_number",
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        "weather_code_synop_4677",
        "weather_code_metar_4678",
        "weather_code_nws",
        "reflectivity_32bit",
        "mor_visibility",
        "sample_interval",
        "laser_amplitude",
        "number_particles",
        "sensor_temperature",
        "sensor_serial_number",
        "firmware_iop",
        "firmware_dsp",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "sensor_status",
        "rain_kinetic_energy",
        "TO_SPLIT",
    ]
    df.columns = names

    # Derive raw drop arrays
    def split_string(s):
        vals = [v.strip() for v in s.split(",")]
        c1 = ", ".join(vals[:32])
        c2 = ", ".join(vals[32:64])
        c3 = ", ".join(vals[64:])
        return pd.Series({"raw_drop_concentration": c1, "raw_drop_average_velocity": c2, "raw_drop_number": c3})

    splitted_string = df["TO_SPLIT"].apply(split_string)
    df["raw_drop_concentration"] = splitted_string["raw_drop_concentration"]
    df["raw_drop_average_velocity"] = splitted_string["raw_drop_average_velocity"]
    df["raw_drop_number"] = splitted_string["raw_drop_number"]

    # Define datetime "time" column
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "station_name",
        "station_number",
        "firmware_iop",
        "firmware_dsp",
        "TO_SPLIT",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
