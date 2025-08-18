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
import os

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file
from disdrodb.utils.logger import log_error


def reader_parsivel(filepath, logger):
    """Reader for Parsivel CR1000 Data Logger file."""
    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
    # - Skip first row as columns names
    reader_kwargs["header"] = None
    # - Skip first 3 rows
    reader_kwargs["skiprows"] = 0
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
    df_raw = read_raw_text_file(
        filepath=filepath,
        column_names=["TO_PARSE"],
        reader_kwargs=reader_kwargs,
        logger=logger,
    )

    # Retrieve header, number of columns and starting rows
    # - Search in the first 3 rows where "TIMESTAMP" occurs
    # - Once identified the row, strip away everything before TIMESTAMP
    # - Then identify start_row_idx as the row where "TIMESTAMP" occurs + 2
    for i in range(3):
        line = df_raw.iloc[i]["TO_PARSE"]
        if "TIMESTAMP" in line:
            # Remove double and single quotes
            line = line.replace('""', '"').replace('"', "")
            # Define header
            timestamp_idx = line.find("TIMESTAMP")
            header_str = line[timestamp_idx:]
            header = header_str.split(",")
            # Define number of columns
            n_columns = len(header)
            # Define start row with data
            start_row_idx = i + 3
            break
    else:
        # start_row_idx = 0
        # n_columns = len(df_raw["TO_PARSE"].iloc[0].split(","))
        raise ValueError("Could not find 'TIMESTAMP' in the first 3 rows of the file.")

    # Retrieve rows with actual data
    df = df_raw.iloc[start_row_idx:]

    # Expand dataframe
    df = df["TO_PARSE"].str.split(",", expand=True, n=n_columns - 1)

    #### Define column names
    if n_columns == 15:
        # 05_VILLENEUVE_DE_BERG_1 (2011)
        # 90_GALABRE (2020)
        column_names = [
            "time",
            "RECORD",
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
            "rain_kinetic_energy",
        ]
    elif n_columns == 16:
        # 33_PRADEL_VIGNES (2011-2015)
        if "Panel_Temp" in header:
            column_names = [
                "time",
                "RECORD",
                "V_Batt_Min",
                "Panel_Temp",
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
            ]
        else:
            # 33_PRADEL_VIGNES (2020)
            column_names = [
                "time",
                "RECORD",
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
                "rain_kinetic_energy",
                "V_Batt_Min",
            ]
    elif n_columns == 19:
        column_names = [
            "time",
            "RECORD",
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
            "sample_interval",
            "sensor_status",
            "rain_kinetic_energy",
            "sensor_temperature_receiver",
            "sensor_temperature_trasmitter",
            "V_Batt_Min",
        ]
    elif n_columns == 20:
        column_names = [
            "time",
            "RECORD",
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
            "sensor_temperature_receiver",
            "sensor_temperature_trasmitter",
            "rain_kinetic_energy",
            "V_Batt_Min",
            "sample_interval",
            "Temps_present",
        ]
    elif n_columns == 24:
        # ALES (2021)
        column_names = [
            "time",  # 0
            "RECORD",  # 1
            "rainfall_rate_32bit",  # 2
            "rainfall_accumulated_32bit",  # 3
            "weather_code_synop_4680",  # 4
            "weather_code_synop_4677",  # 5
            "reflectivity_32bit",  # 6
            "mor_visibility",  # 7
            "laser_amplitude",  # 8
            "number_particles",  # 9
            "sensor_temperature",  # 10
            "sensor_heating_current",  # 11
            "sensor_battery_voltage",  # 12
            "sensor_status",  # # 13
            "rain_kinetic_energy",  # 14
            "AccuH_parsivel",  # 15
            "AccuD_parsivel",  # 16
            "AccuM_parsivel",  # 17
            "AccuY_parsivel",  # 18
            "air_temperature",  # 19
            "relative_humidity",  # 20
            "wind_speed",  # 21
            "wind_direction",  # 22
            "V_Batt_Min",  # 23
        ]
    elif n_columns == 25:
        # AINAC (2024)
        column_names = [
            "time",  # 0
            "RECORD",  # 1
            "rainfall_rate_32bit",  # 2
            "rainfall_accumulated_32bit",  # 3
            "weather_code_synop_4680",  # 4
            "weather_code_synop_4677",  # 5
            "reflectivity_32bit",  # 6
            "mor_visibility",  # 7
            "laser_amplitude",  # 8
            "number_particles",  # 9
            "sensor_temperature",  # 10
            "sensor_heating_current",  # 11
            "sensor_battery_voltage",  # 12
            "sensor_status",  # # 13
            "rain_kinetic_energy",  # 14
            "AccuH_parsivel",  # 15
            "AccuD_parsivel",  # 16
            "AccuM_parsivel",  # 17
            "AccuY_parsivel",  # 18
            "air_temperature",  # 19
            "relative_humidity",  # 20
            "wind_speed",  # 21
            "wind_direction",  # 22
            "V_Batt_Min",  # 23
            "unknown",
        ]
    elif n_columns == 41:
        df = df.iloc[:, :15]
        column_names = [
            "time",  # 0
            "RECORD",  # 1
            "rainfall_rate_32bit",  # 2
            "rainfall_accumulated_32bit",  # 3
            "weather_code_synop_4680",  # 4
            "weather_code_synop_4677",  # 5
            "reflectivity_32bit",  # 6
            "mor_visibility",  # 7
            "laser_amplitude",  # 8
            "number_particles",  # 9
            "sensor_temperature",  # 10
            "sensor_heating_current",  # 11
            "sensor_battery_voltage",  # 12
            "sensor_status",  # # 13
            "rain_kinetic_energy",  # 14
        ]
    elif n_columns == 76:
        # ALES (2009)
        raw_drop_concentration = df.iloc[:, 14:46].agg(",".join, axis=1).str.replace("-10", "0")
        raw_drop_average_velocity = "0,0," + df.iloc[:, 46:].agg(",".join, axis=1)
        df = df.iloc[:, 0:14]
        df["raw_drop_concentration"] = raw_drop_concentration
        df["raw_drop_average_velocity"] = raw_drop_average_velocity

        column_names = [
            "time",
            "RECORD",
            "V_Batt_Min",
            "rainfall_rate_32bit",
            "rainfall_accumulated_32bit",
            "weather_code_synop_4680",
            "weather_code_synop_4677",
            "reflectivity_32bit",
            "mor_visibility",
            "laser_amplitude",
            "number_particles",
            "sensor_heating_current",
            "sensor_serial_numer",
            "error_code",
            "raw_drop_concentration",
            "raw_drop_average_velocity",
        ]

    else:

        raise ValueError(f"{filepath} has {n_columns} columns. Undefined reader.")

    ##------------------------------------------------------------------------.
    #### Assign column names
    df.columns = column_names

    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Define time as datetime column
    df["time"] = pd.to_datetime(df["time"].str.strip('"'), format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Set missing columns as NaN
    potential_missing_columns = [
        "sensor_temperature_receiver",
        "sensor_temperature_trasmitter",
        "rain_kinetic_energy",
    ]
    for column in potential_missing_columns:
        if column not in df.columns:
            df[column] = np.nan

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "RECORD",
        "V_Batt_Min",
        "Panel_Temp",
        "Temps_present",
        "sample_interval",
        "sensor_serial_numer",
        "AccuH_parsivel",
        "AccuD_parsivel",
        "AccuM_parsivel",
        "AccuY_parsivel",
        "unknown",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df


def select_only_valid_rows(df, expected_n_values, logger, filepath):
    """Select only rows with the expected number of values."""
    # Ensure expected_n_values to be a list
    if isinstance(expected_n_values, (int, float)):
        expected_n_values = [expected_n_values]

    # Identify number of values per row
    n_values_per_row = df["TO_PARSE"].apply(lambda x: len(x.split(",")))

    # Get the frequency of each unique number of values
    unique_values, counts = np.unique(n_values_per_row, return_counts=True)

    # Determine the valid number of values
    valid_counts = [(val, count) for val, count in zip(unique_values, counts) if val in expected_n_values]
    if not valid_counts:
        raise ValueError(
            f"{filepath} has no rows with expected number of values: {expected_n_values}."
            f"Found rows with the following number of values: {unique_values}.",
        )

    # Select the most frequent valid number of values
    n_values = max(valid_counts, key=lambda x: x[1])[0]

    # Identify invalid rows
    indices_invalid_values = n_values_per_row != n_values
    invalid_timesteps = df["time"][indices_invalid_values]
    invalid_timesteps_str = list(invalid_timesteps.astype(str))
    # Log if multiple value formats are detected
    if len(unique_values) != 1:
        msg = f"{filepath} has an unexpected number of values at following timesteps: {invalid_timesteps_str}."
        log_error(msg=msg, logger=logger)

    # Remove rows with invalid number of values
    df = df[~indices_invalid_values]

    return df, n_values, invalid_timesteps


def add_nan_at_invalid_timesteps(df, invalid_timesteps):
    """Infill invalid timesteps columns with NaN."""
    # If no invalid timesteps, return dataframe as it is
    if len(invalid_timesteps) == 0:
        return df

    # Create a DataFrame with NaNs and the original time values
    nan_rows = pd.DataFrame({col: ["NaN"] * len(invalid_timesteps) for col in df.columns if col != "time"})
    nan_rows["time"] = invalid_timesteps.to_numpy()

    # Reinsert NaN rows and re-sort by time
    df = pd.concat([df, nan_rows], ignore_index=True).sort_values("time").reset_index(drop=True)
    return df


def reader_spectrum(filepath, logger):
    """Reader for Spectrum CR1000 Data Logger file."""
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
    # - Skip first 3 rows
    reader_kwargs["skiprows"] = 4
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
    # Split and assign columns
    df = df["TO_PARSE"].str.split(",", n=2, expand=True)
    df.columns = ["time", "RECORD", "TO_PARSE"]

    # Define time in datetime format
    df["time"] = pd.to_datetime(df["time"].str.strip('"'), format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Keep only rows with valid number of values
    df, n_values, invalid_timesteps = select_only_valid_rows(
        df=df,
        expected_n_values=[1024, 1054, 1086],
        logger=logger,
        filepath=filepath,
    )

    # Derive raw drop arrays
    if n_values == 1024:
        df["raw_drop_number"] = df["TO_PARSE"]
    elif n_values == 1054:
        # VALESCURE (2014 03-09)
        df_split = df["TO_PARSE"].str.split(",", expand=True)
        raw_drop_average_velocity = "0,0," + df_split.iloc[:, :30].agg(",".join, axis=1)
        raw_drop_number = df_split.iloc[:, 30:].agg(",".join, axis=1)
        df["raw_drop_average_velocity"] = raw_drop_average_velocity
        df["raw_drop_number"] = raw_drop_number
        df["raw_drop_concentration"] = "NaN"
    elif n_values == 1086:
        df_split = df["TO_PARSE"].str.split(",", expand=True)
        raw_drop_concentration = df_split.iloc[:, :32].agg(",".join, axis=1).str.replace("-10", "0")
        raw_drop_average_velocity = "0,0," + df_split.iloc[:, 32:62].agg(",".join, axis=1)
        raw_drop_number = df_split.iloc[:, 62:].agg(",".join, axis=1)
        df["raw_drop_concentration"] = raw_drop_concentration
        df["raw_drop_average_velocity"] = raw_drop_average_velocity
        df["raw_drop_number"] = raw_drop_number
    else:
        raise ValueError(f"{filepath} has {n_values} spectrum values. Undefined reader.")

    # Drop columns not agreeing with DISDRODB L0 standards
    df = df.drop(columns=["TO_PARSE", "RECORD"])

    # Infill with NaN at invalid timesteps
    add_nan_at_invalid_timesteps(df, invalid_timesteps)
    return df


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    # Retrieve spectrum filepath
    spectrum_filepath = filepath.replace("parsivel", "spectre")

    # Read integral variables
    df = reader_parsivel(filepath, logger=logger)

    # Drop duplicates timesteps
    df = df.drop_duplicates(subset="time", keep="first")

    # Initialize empty arrays
    # --> 0 values array produced in L0B
    arrays_columns = ["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number"]
    for c in arrays_columns:
        if c not in df:
            df[c] = ""

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
