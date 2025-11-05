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
"""DISDRODB reader for GID LPM sensors not measuring wind."""
import os

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file
from disdrodb.utils.logger import log_warning


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    ##------------------------------------------------------------------------.
    #### - Define raw data headers
    column_names = ["TO_PARSE"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    # - For more info: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # Since column names are expected to be passed explicitly, header is set to None
    reader_kwargs["header"] = None

    # - Number of rows to be skipped at the beginning of the file
    reader_kwargs["skiprows"] = None

    # - Define encoding
    reader_kwargs["encoding"] = "latin"

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
    # Raise error if empty file
    if len(df) == 0:
        raise ValueError(f"{filepath} is empty.")

    # Select only rows with expected number of delimiters
    df = df[df["TO_PARSE"].str.count(";").isin([519, 520])]

    # Check there are still valid rows
    if len(df) == 0:
        raise ValueError(f"No valid rows in {filepath}.")

    # Split by ; delimiter (before raw drop number)
    df = df["TO_PARSE"].str.split(";", expand=True, n=79)

    # Assign column names
    names = [
        "start_identifier",
        "device_address",
        "sensor_serial_number",
        "sensor_date",
        "sensor_time",
        "weather_code_synop_4677_5min",
        "weather_code_synop_4680_5min",
        "weather_code_metar_4678_5min",
        "precipitation_rate_5min",
        "weather_code_synop_4677",
        "weather_code_synop_4680",
        "weather_code_metar_4678",
        "precipitation_rate",
        "rainfall_rate",
        "snowfall_rate",
        "precipitation_accumulated",
        "mor_visibility",
        "reflectivity",
        "quality_index",
        "max_hail_diameter",
        "laser_status",
        "static_signal_status",
        "laser_temperature_analog_status",
        "laser_temperature_digital_status",
        "laser_current_analog_status",
        "laser_current_digital_status",
        "sensor_voltage_supply_status",
        "current_heating_pane_transmitter_head_status",
        "current_heating_pane_receiver_head_status",
        "temperature_sensor_status",
        "current_heating_voltage_supply_status",
        "current_heating_house_status",
        "current_heating_heads_status",
        "current_heating_carriers_status",
        "control_output_laser_power_status",
        "reserved_status",
        "temperature_interior",
        "laser_temperature",
        "laser_current_average",
        "control_voltage",
        "optical_control_voltage_output",
        "sensor_voltage_supply",
        "current_heating_pane_transmitter_head",
        "current_heating_pane_receiver_head",
        "temperature_ambient",
        "current_heating_voltage_supply",
        "current_heating_house",
        "current_heating_heads",
        "current_heating_carriers",
        "number_particles",
        "number_particles_internal_data",
        "number_particles_min_speed",
        "number_particles_min_speed_internal_data",
        "number_particles_max_speed",
        "number_particles_max_speed_internal_data",
        "number_particles_min_diameter",
        "number_particles_min_diameter_internal_data",
        "number_particles_no_hydrometeor",
        "number_particles_no_hydrometeor_internal_data",
        "number_particles_unknown_classification",
        "number_particles_unknown_classification_internal_data",
        "number_particles_class_1",
        "number_particles_class_1_internal_data",
        "number_particles_class_2",
        "number_particles_class_2_internal_data",
        "number_particles_class_3",
        "number_particles_class_3_internal_data",
        "number_particles_class_4",
        "number_particles_class_4_internal_data",
        "number_particles_class_5",
        "number_particles_class_5_internal_data",
        "number_particles_class_6",
        "number_particles_class_6_internal_data",
        "number_particles_class_7",
        "number_particles_class_7_internal_data",
        "number_particles_class_8",
        "number_particles_class_8_internal_data",
        "number_particles_class_9",
        "number_particles_class_9_internal_data",
        "raw_drop_number",
    ]
    df.columns = names

    # Remove checksum from raw_drop_number
    df["raw_drop_number"] = df["raw_drop_number"].str.strip(";").str.rsplit(";", n=1, expand=True)[0]

    # Identify rows with bad sensor date (compared to file name)
    filename = os.path.basename(filepath)
    file_date_str = filename[0:8]
    idx_bad_date = df["sensor_date"] != pd.to_datetime(file_date_str, format="%Y%m%d").strftime("%d.%m.%y")

    # If all rows have bad sensor_date, use the one of the file name
    if idx_bad_date.all():
        # -  Infer and define "time" column
        start_time_str = filename[0:10]
        start_time = pd.to_datetime(start_time_str, format="%Y%m%d%H")

        # - Define timedelta based on sensor_time
        # --> Add +24h to subsequent times when time resets
        dt = pd.to_timedelta(df["sensor_time"]).to_numpy().astype("m8[s]")
        rollover_indices = np.where(np.diff(dt) < np.timedelta64(0, "s"))[0]
        if rollover_indices.size > 0:
            for idx in rollover_indices:
                dt[idx + 1 :] += np.timedelta64(24, "h")
        dt = dt - dt[0]

        # - Define approximate time
        df["time"] = start_time + dt

        # - Keep rows where time increment is between 00 and 59 minutes
        valid_rows = dt <= np.timedelta64(3540, "s")
        df = df[valid_rows]

    # If only some rows have bad sensor date, just drop such bad rows
    else:
        if idx_bad_date.any():
            # Remove rows with bad dates
            bad_dates = df[idx_bad_date]["sensor_date"].unique().tolist()
            df = df[~idx_bad_date]
            msg = f"{filepath} contain rows with the following unexpected sensor_date values {bad_dates}"
            log_warning(msg=msg, logger=logger)

        # Define datetime "time" column
        df["time"] = df["sensor_date"] + "-" + df["sensor_time"]
        df["time"] = pd.to_datetime(df["time"], format="%d.%m.%y-%H:%M:%S", errors="coerce")

    # Drop row if start_identifier different than 00
    df = df[df["start_identifier"].astype(str) == "00"]

    # Drop rows with invalid raw_drop_number
    df = df[df["raw_drop_number"].astype(str).str.len() == 1759]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "start_identifier",
        "device_address",
        "sensor_serial_number",
        "sensor_date",
        "sensor_time",
    ]
    df = df.drop(columns=columns_to_drop)
    return df
