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
"""DISDRODB reader for TU Wien PWS100 raw text data."""
import os

import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file


def reader_spectrum(
    filepath,
    logger=None,
):
    """Reader spectrum file."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = ["TO_SPLIT"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"

    # - Skip first row as columns names
    reader_kwargs["header"] = None

    # - Skip header
    reader_kwargs["skiprows"] = 4

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
    reader_kwargs["na_values"] = ["na", "error", "-.-", " NA", "NAN"]

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
    # Remove corrupted rows (and header)
    df = df[df["TO_SPLIT"].str.count(",") == 1157]

    # Split into columns
    df = df["TO_SPLIT"].str.split(",", expand=True, n=2)

    # Assign columns names
    names = [
        "time",
        "record",
        "raw_drop_number",  # "Size_0.00_Vel_0.00","Size_0.00_Vel_0.10", ...
    ]
    df.columns = names

    # Add datetime time column
    df["time"] = df["time"].str.replace('"', "")
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Clean raw_drop_number '"NAN"' --> 'NaN'
    df["raw_drop_number"] = df["raw_drop_number"].str.replace('"NAN"', "NaN")

    # Drop columns not needed
    df = df.drop(columns=["record"])
    return df


def reader_met_file(filepath, logger):
    """Reader MET file."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = ["TO_SPLIT"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"

    # - Skip first row as columns names
    reader_kwargs["header"] = None

    # - Skip header
    reader_kwargs["skiprows"] = 4

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
    reader_kwargs["na_values"] = ["na", "error", "-.-", " NA", "NAN"]

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
    df = df[df["TO_SPLIT"].str.count(",") == 40]

    # Split into columns
    df = df["TO_SPLIT"].str.split(",", expand=True)

    # Assign columns names
    names = [
        "time",
        "RECORD",
        "PWS100_Year",
        "PWS100_Month",
        "PWS100_Day",
        "PWS100_Hours",
        "PWS100_Minutes",
        "PWS100_Seconds",
        "mor_visibility",
        "weather_code_synop_4680",
        "weather_code_metar_4678",
        "weather_code_nws",
        "PWS100_PWCode_NWS_String",
        "air_temperature",
        "relative_humidity",
        "air_temperature_min",
        "air_temperature_max",
        "rainfall_rate",
        "rainfall_accumulated",
        "average_drop_velocity",
        "average_drop_size",
        "PWS100_PartType_Drizzle",
        "PWS100_PartType_FreezingDrizzle",
        "PWS100_PartType_Rain",
        "PWS100_PartType_FreezingRain",
        "PWS100_PartType_SnowGrains",
        "PWS100_PartType_SnowFlakes",
        "PWS100_PartType_IcePellets",
        "PWS100_PartType_Hail",
        "PWS100_PartType_Graupel",
        "PWS100_PartType_Error",
        "PWS100_PartType_Unknown",
        "PWS100_VISAlarm1",
        "PWS100_VISAlarm2",
        "PWS100_VISAlarm3",
        "PWS100_CleanLaserWindow",
        "PWS100_CleanUpperWindow",
        "PWS100_CleanLowerWindow",
        "sensor_status",
        "PWS100_FaultStatus_EN",
        "PWS100_PowerStatus",
    ]
    df.columns = names

    # Remove rows with only NaN
    df = df[df["PWS100_Year"] != '"NAN"']

    # Define type distribution variable
    type_distribution_columns = [
        "PWS100_PartType_Drizzle",
        "PWS100_PartType_FreezingDrizzle",
        "PWS100_PartType_Rain",
        "PWS100_PartType_FreezingRain",
        "PWS100_PartType_SnowGrains",
        "PWS100_PartType_SnowFlakes",
        "PWS100_PartType_IcePellets",
        "PWS100_PartType_Hail",
        "PWS100_PartType_Graupel",
        "PWS100_PartType_Error",
        "PWS100_PartType_Unknown",
    ]
    df["type_distribution"] = df[type_distribution_columns].agg(",".join, axis=1)

    # Define alarms
    # - should be 16 values
    # alarms_columns = [
    #     "PWS100_VISAlarm1",
    #     "PWS100_VISAlarm2",
    #     "PWS100_VISAlarm3",
    #     "PWS100_CleanLaserWindow",
    #     "PWS100_CleanUpperWindow",
    #     "PWS100_CleanLowerWindow",
    #     "PWS100_FaultStatus",
    #     "PWS100_FaultStatus_EN",
    #     "PWS100_PowerStatus",
    # ]
    # df["alarms"] = df[alarms_columns].agg(",".join, axis=1)

    # Define datetime "time" column from filename
    df["time"] = df["time"].str.replace('"', "")
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")

    # # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "RECORD",
        "PWS100_Year",
        "PWS100_Month",
        "PWS100_Day",
        "PWS100_Hours",
        "PWS100_Minutes",
        "PWS100_Seconds",
        "PWS100_PartType_Drizzle",
        "PWS100_PartType_FreezingDrizzle",
        "PWS100_PartType_Rain",
        "PWS100_PartType_FreezingRain",
        "PWS100_PartType_SnowGrains",
        "PWS100_PartType_SnowFlakes",
        "PWS100_PartType_IcePellets",
        "PWS100_PartType_Hail",
        "PWS100_PartType_Graupel",
        "PWS100_PartType_Error",
        "PWS100_PartType_Unknown",
        "PWS100_VISAlarm1",
        "PWS100_VISAlarm2",
        "PWS100_VISAlarm3",
        "PWS100_CleanLaserWindow",
        "PWS100_CleanUpperWindow",
        "PWS100_CleanLowerWindow",
        "PWS100_FaultStatus_EN",
        "PWS100_PowerStatus",
        "PWS100_PWCode_NWS_String",
    ]
    df = df.drop(columns=columns_to_drop)
    return df


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    # Retrieve spectrum filepath
    spectrum_filepath = filepath.replace("WS_MET_PWS100_Data", "WS_MET_Size_Vel_distr")

    # Read integral variables
    df = reader_met_file(filepath, logger=logger)

    # Drop duplicates timesteps
    df = df.drop_duplicates(subset="time", keep="first")

    # Initialize raw_drop_number array
    # --> 0 values array produced in L0B
    df["raw_drop_number"] = ""

    # Add raw spectrum if available
    if os.path.exists(spectrum_filepath):
        # Read raw spectrum for corresponding timesteps
        df_raw_spectrum = reader_spectrum(spectrum_filepath, logger=logger)
        df_raw_spectrum = df_raw_spectrum.drop_duplicates(subset="time", keep="first")
        # Add raw array to df
        df = df.set_index("time")
        df_raw_spectrum = df_raw_spectrum.set_index("time")
        df.update(df_raw_spectrum)
        # Set back time as column
        df = df.reset_index()

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
