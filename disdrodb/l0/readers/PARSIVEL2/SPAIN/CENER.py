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
    column_names = ["TO_PARSE"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
    # - Skip first row as columns names
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
    df.columns = ["date", "time", "TO_PARSE"]
    datetime_str = df["date"] + " " + df["time"]
    df["time"] = pd.to_datetime(datetime_str, format="%d.%m.%Y %H:%M:%S", errors="coerce")

    # Identify rows with integral variables
    df_vars = df[df["TO_PARSE"].str.len() == 94]

    # Split and assign column names
    df_data = df_vars["TO_PARSE"].str.split(",", expand=True)
    var_names = [
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
        "sensor_serial_number",
        "sensor_temperature_receiver",
        "sensor_temperature_trasmitter",
        "snowfall_rate",
        "rain_kinetic_energy",
    ]
    df_data.columns = var_names
    df_data["time"] = df_vars["time"]
    df_data = df_data.drop(columns="sensor_serial_number")

    # Initialize empty arrays
    # --> 0 values array produced in L0B
    df_data["raw_drop_concentration"] = ""
    df_data["raw_drop_average_velocity"] = ""
    df_data["raw_drop_number"] = ""

    # Identify raw spectrum
    df_raw_spectrum = df[df["TO_PARSE"].str.len() == 4545]

    # Derive raw drop arrays
    def split_string(s):
        vals = [v.strip() for v in s.split(",")]
        c1 = ",".join(vals[:32])
        c2 = ",".join(vals[32:64])
        c3 = ",".join(vals[64].replace("r", "").split("/"))
        series = pd.Series(
            {
                "raw_drop_concentration": c1,
                "raw_drop_average_velocity": c2,
                "raw_drop_number": c3,
            },
        )
        return series

    splitted_string = df_raw_spectrum["TO_PARSE"].apply(split_string)
    df_raw_spectrum["raw_drop_concentration"] = splitted_string["raw_drop_concentration"]
    df_raw_spectrum["raw_drop_average_velocity"] = splitted_string["raw_drop_average_velocity"]
    df_raw_spectrum["raw_drop_number"] = splitted_string["raw_drop_number"]
    df_raw_spectrum = df_raw_spectrum.drop(columns=["date", "TO_PARSE"])

    # Add raw array
    df = df_data.set_index("time")
    df_raw_spectrum = df_raw_spectrum.set_index("time")

    df.update(df_raw_spectrum)

    # Set back time as column
    df = df.reset_index()

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
