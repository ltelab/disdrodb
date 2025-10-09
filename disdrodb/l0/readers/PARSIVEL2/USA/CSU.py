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
"""DISDRODB reader for Colorado State University OTT Parsivel 2 raw data."""

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
    df = df[df["TO_PARSE"].str.count(",").isin([15, 1040])]

    # Split into columns
    df["TO_PARSE"] = df["TO_PARSE"] + ",0"
    df = df["TO_PARSE"].str.split(",", expand=True, n=16)
    
    # Assign columns names
    names = [
        "date",
        "time",
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        "weather_code_metar_4678",   
        "weather_code_nws",  
        "reflectivity_32bit",
        "mor_visibility",
        "laser_amplitude",
        "number_particles",
        "sensor_temperature",
        "sensor_heating_current",
        "sensor_battery_voltage", 
        "rain_kinetic_energy",    
        "snowfall_rate", 
        "raw_drop_number",
    ]
    df.columns = names   

    # Add datetime time column
    time_str = df["date"] + "-" + df["time"]
    df["time"] = pd.to_datetime(time_str, format="%d.%m.%Y-%H:%M:%S", errors="coerce")
    
    # Derive the raw spectrum
    # - When no drops detected, None
    # - After conversion to string, becomes NaN
    df["raw_drop_number"] = df["raw_drop_number"].astype("string")
    df["raw_drop_number"] = df["raw_drop_number"].str.strip()
    
    # Remove <SPECTRUM> and </SPECTRUM> prefix and suffix from the raw_drop_number field
    df["raw_drop_number"] = df["raw_drop_number"].str.replace("<SPECTRUM>", "")
    df["raw_drop_number"] = df["raw_drop_number"].str.replace("</SPECTRUM>,0", "")

    # Preprocess the raw spectrum and raw_drop_average_velocity
    # - Add 0 before every ; if ; not preceded by a digit
    # - Example: ';;1;;' --> '0;0;1;0;'
    df["raw_drop_number"] = df["raw_drop_number"].str.replace(r"(?<!\d),", "0,", regex=True)
    
    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "date"
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
