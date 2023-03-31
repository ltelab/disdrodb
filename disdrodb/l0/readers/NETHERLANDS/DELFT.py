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
    column_names = ["time", "epoch_time", "TO_BE_PARSED"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ";"
    # - Skip first row as columns names
    reader_kwargs["header"] = None
    # - Avoid first column to become df index
    reader_kwargs["index_col"] = False
    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"
    # - Define parser engine
    #   - C engine is faster
    #   - Python engine is more feature-complete
    reader_kwargs["engine"] = "python"
    # - Define on-the-fly decompression of on-disk data
    #   - Available: gzip, bz2, zip
    reader_kwargs["compression"] = "infer"
    # reader_kwargs['zipped'] = False
    # reader_kwargs['zipped'] = True
    # - Strings to recognize as NA/NaN and replace with standard NA flags
    #   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
    #                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
    #                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
    reader_kwargs["na_values"] = ["na", "", "error", "-.-", " NA"]
    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    # Station 8 has all raw_drop_number corrupted, so it can't be used

    def df_sanitizer_fun(df):
        # Import pandas
        import pandas as pd

        # - Remove rows that have a corrupted "TO_BE_PARSED" column
        # - PAR001, PAR002 have length 3726 (no station_name)
        # - PAR007 have length 3736 ()
        df = df.loc[df["TO_BE_PARSED"].astype(str).str.len() >= 3726]
        df = df.loc[df["TO_BE_PARSED"].astype(str).str.len() <= 3736]

        # - Convert 'time' column to datetime
        df_time = pd.to_datetime(df["time"], format="%Y%m%d-%H%M%S", errors="coerce")

        # - Strip values from start and end of the string
        df["TO_BE_PARSED"] = df["TO_BE_PARSED"].str.lstrip("b'").str.rstrip("'").str.rstrip("\\r\\n'")

        # - Split the column 'TO_BE_PARSED'
        df_to_parse = df["TO_BE_PARSED"].str.split(";", expand=True, n=99)

        # - Retrieve DISDRODB compliant columns
        df = df_to_parse.iloc[:, 0:35]

        # - Assign column names
        column_names = [
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
            "sensor_time_measurement_start",
            "sensor_time",
            "sensor_date",
            "station_name",
            "station_number",
            "rainfall_amount_absolute_32bit",
            "error_code",
            "sensor_temperature_pcb",
            "sensor_temperature_receiver",
            "sensor_temperature_trasmitter",
            "rainfall_rate_16_bit_30",
            "rainfall_rate_16_bit_1200",
            "rainfall_accumulated_16bit",
            "reflectivity_16bit",
            "rain_kinetic_energy",
            "snowfall_rate",
            "number_particles_all",
            # "number_particles_all_detected",
        ]
        df.columns = column_names

        # - Add time column
        df["time"] = df_time

        # - Retrieve raw_drop_concentration
        df["raw_drop_concentration"] = df_to_parse.iloc[:, 35:67].apply(
            lambda x: ",".join(x.dropna().astype(str)),
            axis=1,
        )
        # - Retrieve raw_drop_average_velocity
        df["raw_drop_average_velocity"] = df_to_parse.iloc[:, 67:99].apply(
            lambda x: ",".join(x.dropna().astype(str)),
            axis=1,
        )

        # - Retrieve raw_drop_number
        df_raw_drop_number = df_to_parse.iloc[:, 99].squeeze()
        df_raw_drop_number = df_raw_drop_number.str.replace(r"(\w{3})", r"\1,", regex=True)
        df["raw_drop_number"] = df_raw_drop_number

        # - Drop columns not agreeing with DISDRODB L0 standards
        columns_to_drop = [
            "firmware_iop",
            "firmware_dsp",
            "sensor_time_measurement_start",
            "sensor_time",
            "sensor_date",
            "station_name",
            "station_number",
            "sensor_serial_number",
            "sample_interval",
            "sensor_serial_number",
            # "epoch_time",
            # "number_particles_all_detected",
        ]
        df = df.drop(columns=columns_to_drop)

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_name>
    glob_patterns = "*/*/*.csv"  # <year>/<yearmonth>/*csv

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
