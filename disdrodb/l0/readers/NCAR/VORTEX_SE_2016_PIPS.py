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
    column_names = [
        "logger_time",
        "record_number",
        "battery_voltage",
        "logger_temperature",
        "wind_direction",
        "wind_speed",
        "wind_flag",
        "fast_temperature",
        "slow_temperature",
        "relative_humidity",
        "pressure",
        "compass_direction",
        "gps_time",
        "gps_status",
        "gps_latitude",  # DD.DM, DM = decimal minutes/100, DD = degrees
        "gps_latitude_hemisphere",  # N/S
        "gps_longitude",
        "gps_longitude_hemisphere",  # W/E
        "gps_speed",
        "gps_direction",
        "gps_date",
        "gps_magnetic_version",
        "gps_altitude",
        "relative_wind_direction",
        "dew_point_temperature",
        "rh",
        "disdrometer_data",
    ]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ","

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"

    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"

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
    reader_kwargs["na_values"] = ["na", "", "error"]

    # Skip first row as columns names
    reader_kwargs["header"] = None

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    def df_sanitizer_fun(df):
        # - Import pandas
        import pandas as pd

        # - Drop timesteps without disdrometer data
        df = df[~df["disdrometer_data"].isna()]

        # - Retrieve disdrometer data
        df_data = df["disdrometer_data"].str.split(";", expand=True, n=11)
        # - Assign column names
        column_names = [
            "serial_number",
            "rainfall_rate_32bit",
            "rainfall_accumulated_32bit",
            "reflectivity_32bit",
            "sample_interval",
            "laser_amplitude",
            "number_particles",
            "sensor_temperature",
            "sensor_battery_voltage",
            "sensor_time",  # Note: logger_time is currently used !
            "sensor_date",
            "raw_drop_number",
        ]
        df_data.columns = column_names

        # - Retrieve time and coordinates information
        # --> Latitude in degrees_north
        # --> Longitude in degrees_east
        df_time = pd.to_datetime(df["logger_time"], errors="coerce")
        df_lat_sign = df["gps_latitude_hemisphere"].str.replace("N", "1").str.replace("S", "-1")
        df_lon_sign = df["gps_longitude_hemisphere"].str.replace("E", "1").str.replace("W", "-1")
        df_lat_sign = df_lat_sign.astype(float)
        df_lon_sign = df_lon_sign.astype(float)
        df_lon = df["gps_longitude"].astype(float)
        df_lat = df["gps_latitude"].astype(float)
        df_lon = df_lon * df_lon_sign
        df_lat = df_lat * df_lat_sign

        # Create dataframe
        df = df_data
        df["time"] = df_time
        df["latitude"] = df_lat
        df["longitude"] = df_lon

        # - Drop columns not agreeing with DISDRODB L0 standards
        df = df.drop(columns=["serial_number", "sensor_time", "serial_number"])

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
