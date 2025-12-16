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
"""Reader for the EROSION campaign in Denmark."""

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file

COLUMNS = [
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
    "sensor_heating_current",
    "sensor_battery_voltage",
    "sensor_status",
    "rain_kinetic_energy",
    "snowfall_rate",
    "raw_drop_concentration",
    "raw_drop_average_velocity",
    "raw_drop_number",
]


def read_par_format(filepath, logger):
    """Read .par data format."""
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
    reader_kwargs["skiprows"] = 1

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
    df_raw = read_raw_text_file(
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        logger=logger,
    )

    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    n_separators, counts = np.unique(df_raw["TO_PARSE"].str.count(","), return_counts=True)
    n_separators = n_separators[counts.argmax()]

    # Assign names
    if n_separators == 1113:
        nsplit = 25
        names = [
            "id",
            "y",
            "m",
            "d",
            "hh",
            "mm",
            "ss",
            "rainfall_accumulated_32bit",
            "rainfall_rate_32bit",
            "snowfall_rate",
            "reflectivity_32bit",
            "rain_kinetic_energy",
            "mor_visibility",
            "weather_code_synop_4680",
            "weather_code_synop_4677",
            "weather_code_metar_4678",
            # "weather_code_nws",
            "firmware_iop",
            "firmware_dsp",
            "sensor_status",
            "htst",
            "sensor_temperature",
            "sensor_battery_voltage",
            "laser_amplitude",
            "number_particles",
            "nPART",
            "TO_SPLIT",
        ]
    elif n_separators == 1114:
        nsplit = 26
        names = [
            "id",
            "y",
            "m",
            "d",
            "hh",
            "mm",
            "ss",
            "rainfall_accumulated_32bit",
            "rainfall_rate_32bit",
            "snowfall_rate",
            "reflectivity_32bit",
            "rain_kinetic_energy",
            "mor_visibility",
            "weather_code_synop_4680",
            "weather_code_synop_4677",
            "weather_code_metar_4678",
            "weather_code_nws",
            "firmware_iop",
            "firmware_dsp",
            "sensor_status",
            "htst",
            "sensor_temperature",
            "sensor_battery_voltage",
            "laser_amplitude",
            "number_particles",
            "nPART",
            "TO_SPLIT",
        ]
    else:
        raise NotImplementedError("Unrecognized number of columns")

    # Remove corrupted rows
    df_raw = df_raw[df_raw["TO_PARSE"].str.count(",") == n_separators]

    # Create ID and Value columns
    df = df_raw["TO_PARSE"].str.split(",", expand=True, n=nsplit)

    # Assign names
    df.columns = names

    # Define datetime "time" column
    df["time"] = pd.to_datetime(
        {"year": df["y"], "month": df["m"], "day": df["d"], "hour": df["hh"], "minute": df["mm"], "second": df["ss"]},
    )

    # Retrieve raw array
    df_split = df["TO_SPLIT"].str.split(",", expand=True)
    df["raw_drop_concentration"] = df_split.iloc[:, :32].agg(",".join, axis=1)
    df["raw_drop_average_velocity"] = df_split.iloc[:, 32:64].agg(",".join, axis=1)
    df["raw_drop_number"] = df_split.iloc[:, 64:].agg(",".join, axis=1)
    df["raw_drop_number"] = df["raw_drop_number"].str.replace("-9", "0")
    del df_split

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "nPART",
        "htst",
        "id",
        "y",
        "m",
        "d",
        "hh",
        "mm",
        "ss",
        "firmware_iop",
        "firmware_dsp",
        "TO_SPLIT",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df


def read_asdo_format(filepath, logger):
    """Read ASDO format."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = ["TO_PARSE"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = None

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
    reader_kwargs["engine"] = "c"

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
    df = df["TO_PARSE"].str.split(":", expand=True, n=1)
    df.columns = ["ID", "Value"]

    # Select only rows with values
    df = df[df["Value"].astype(bool)]
    df = df[df["Value"].apply(lambda x: x is not None)]

    # Drop rows with invalid IDs
    # - Corrupted rows
    valid_id_str = np.char.rjust(np.arange(0, 98).astype(str), width=2, fillchar="0")
    df = df[df["ID"].astype(str).isin(valid_id_str)]

    # Raise error if no more rows after removed corrupted ones
    if len(df) == 0:
        raise ValueError("No rows left after removing corrupted ones.")

    # Create the dataframe with each row corresponding to a timestep
    # group -> row, ID -> column
    df["_group"] = (df["ID"].astype(int).diff() <= 0).cumsum()
    df = df.pivot(index="_group", columns="ID")  # noqa
    df.columns = df.columns.get_level_values("ID")
    df = df.reset_index(drop=True)

    # Define column names
    column_dict = {
        "01": "rainfall_rate_32bit",
        "02": "rainfall_accumulated_32bit",
        "03": "weather_code_synop_4680",
        "04": "weather_code_synop_4677",
        "05": "weather_code_metar_4678",
        "06": "weather_code_nws",
        "07": "reflectivity_32bit",
        "08": "mor_visibility",
        "09": "sample_interval",
        "10": "laser_amplitude",
        "11": "number_particles",
        "12": "sensor_temperature",
        # "13": "sensor_serial_number",
        # "14": "firmware_iop",
        # "15": "firmware_dsp",
        "16": "sensor_heating_current",
        "17": "sensor_battery_voltage",
        "18": "sensor_status",
        "19": "start_time",
        "20": "sensor_time",
        "21": "sensor_date",
        # "22": "station_name",
        # "23": "station_number",
        # "24": "rainfall_amount_absolute_32bit",
        # "25": "error_code",
        # "26": "sensor_temperature_pcb",
        # "27": "sensor_temperature_receiver",
        # "28": "sensor_temperature_trasmitter",
        # "30": "rainfall_rate_16_bit_30",
        # "31": "rainfall_rate_16_bit_1200",
        # "32": "rainfall_accumulated_16bit",
        "34": "rain_kinetic_energy",
        "35": "snowfall_rate",
        "90": "raw_drop_concentration",
        "91": "raw_drop_average_velocity",
        "93": "raw_drop_number",
    }

    # Identify missing columns and add NaN
    missing_columns = COLUMNS[np.isin(COLUMNS, df.columns, invert=True)].tolist()
    if len(missing_columns) > 0:
        for column in missing_columns:
            df[column] = "NaN"

    # Rename columns
    df = df.rename(column_dict, axis=1)

    # Keep only columns defined in the dictionary
    df = df[list(column_dict.values())]

    # Define datetime "time" column
    df["time"] = df["sensor_date"] + "-" + df["sensor_time"]
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y-%H:%M:%S", errors="coerce")

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "sensor_date",
        "sensor_time",
        # "firmware_iop",
        # "firmware_dsp",
        # "sensor_serial_number",
        # "station_name",
        # "station_number",
    ]
    df = df.drop(columns=columns_to_drop)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    # Choose the appropriate reader based on the file extension
    if filepath.endswith(".par"):  # e.g. in Thyboron  # noqa: SIM108
        df = read_par_format(filepath, logger)
    else:  # atm4
        df = read_asdo_format(filepath, logger)

    # Identify missing columns and add NaN
    expected_columns = np.array(COLUMNS)
    missing_columns = expected_columns[np.isin(expected_columns, df.columns, invert=True)].tolist()
    if len(missing_columns) > 0:
        for column in missing_columns:
            df[column] = "NaN"

    # Return the dataframe adhering to DISDRODB L0 standards
    return df
