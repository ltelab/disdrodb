#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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
from disdrodb.l0 import run_l0a
from disdrodb.l0.l0_reader import reader_generic_docstring, is_documented_by


@is_documented_by(reader_generic_docstring)
def reader(
    raw_dir,
    processed_dir,
    station_name,
    # Processing options
    force=False,
    verbose=False,
    parallel=False,
    debugging_mode=False,
):
    ##------------------------------------------------------------------------.
    #### - Define column names
    column_names = ["TO_BE_PARSED"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = "/\n"

    # Skip first row as columns names
    reader_kwargs["header"] = None

    # Skip first 2 rows
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
    #   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
    #                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
    #                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
    reader_kwargs["na_values"] = ["na", "", "error", "NA"]

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer

    def df_sanitizer_fun(df):
        # Import numpy and pandas
        import numpy as np
        import pandas as pd

        # Remove rows with unvalid length
        # - time: 20
        # - data: 4638
        df = df[df["TO_BE_PARSED"].str.len().isin([20, 4638])]
        df = df.reset_index(drop=True)

        # Remove rows with consecutive timesteps
        # - Keep last timestep occurence
        idx_timesteps = np.where(df["TO_BE_PARSED"].str.len() == 20)[0]
        idx_without_data = np.where(np.diff(idx_timesteps) == 1)[0].flatten().astype(int)
        idx_timesteps_without_data = idx_timesteps[idx_without_data]
        df = df.drop(labels=idx_timesteps_without_data)

        # If the last row is a timestep, remove it
        if df["TO_BE_PARSED"].str.len().iloc[-1] == 20:
            df = df[:-1]

        # Check there are data to process
        if len(df) == 0 or len(df) == 1:
            raise ValueError("No data to process.")

        # Retrieve time
        df_time = df[::2]
        df_time = df_time.reset_index(drop=True)

        # Retrieve data
        df_data = df[1::2]
        df_data = df_data.reset_index(drop=True)

        if len(df_time) != len(df_data):
            raise ValueError("Likely corrupted data. Not same number of timesteps and data.")

        # Remove starting - from timestep
        df_time = df_time["TO_BE_PARSED"].str.replace("-", "", n=1)

        # Create dataframe
        df_data["time"] = df_time.to_numpy()

        # Count number of delimiters to identify valid rows
        df_data = df_data[df_data["TO_BE_PARSED"].str.count(";") == 1104]

        # Split by ; delimiter
        df = df_data["TO_BE_PARSED"].str.split(";", expand=True, n=16)

        # Assign column names
        column_names = [
            "rainfall_rate_32bit",
            "rainfall_accumulated_32bit",
            "weather_code_synop_4680",
            "weather_code_synop_4677",
            "weather_code_metar_4678",
            "reflectivity_32bit",
            "mor_visibility",
            "laser_amplitude",
            "number_particles",
            "unknown1",
            "sensor_battery_voltage",
            "sensor_status",
            "station_name",
            "sensor_temperature",
            "rainfall_amount_absolute_32bit",
            "error_code",
            "RAW_TO_PARSE",
        ]

        df.columns = column_names
        # Add valid timestep
        df["time"] = pd.to_datetime(df_data["time"], format="%Y-%m-%d %H:%M:%S")

        # Add raw array
        df["raw_drop_concentration"] = df["RAW_TO_PARSE"].str[:224]
        df["raw_drop_average_velocity"] = df["RAW_TO_PARSE"].str[224:448]
        df["raw_drop_number"] = df["RAW_TO_PARSE"].str[448:]

        # Drop columns not agreeing with DISDRODB L0 standards
        df = df.drop(columns=["station_name", "RAW_TO_PARSE", "unknown1"])

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_name>
    glob_patterns = "*.txt"

    ####----------------------------------------------------------------------.
    #### - Create L0A products
    run_l0a(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        station_name=station_name,
        # Custom arguments of the reader for L0A processing
        glob_patterns=glob_patterns,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
        # Processing options
        force=force,
        verbose=verbose,
        parallel=parallel,
        debugging_mode=debugging_mode,
    )
