#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:41:27 2022

@author: kimbo
"""

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

from disdrodb.L0 import run_L0


from disdrodb.L0.L0_processing import reader_generic_docstring, is_documented_by


@is_documented_by(reader_generic_docstring)
def reader(
    raw_dir,
    processed_dir,
    l0a_processing=True,
    l0b_processing=True,
    keep_l0a=False,
    force=False,
    verbose=False,
    debugging_mode=False,
    lazy=True,
    single_netcdf=True,
):

    ####----------------------------------------------------------------------.
    ###########################
    #### CUSTOMIZABLE CODE ####
    ###########################
    #### - Define raw data headers
    # Notes
    # - In all files, the datalogger voltage hasn't the delimeter,
    #   so need to be split to obtain datalogger_voltage and rainfall_rate_32bit
    column_names = ["temp"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"

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
    reader_kwargs["na_values"] = ["na", "", "error"]

    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    # Skip first row as columns names
    reader_kwargs["header"] = None

    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"

    ##------------------------------------------------------------------------.
    #### - Define facultative dataframe sanitizer function for L0 processing
    # - Enable to deal with bad raw data files
    # - Enable to standardize raw data files to L0 standards  (i.e. time to datetime)
    df_sanitizer_fun = None

    def df_sanitizer_fun(df, lazy=False):
        # Import dask or pandas
        # No lazy mode for now
        if lazy:
            import pandas as dd

            df = df.compute()
        else:
            import pandas as dd

        # Reshape dataframe
        a = df.to_numpy()
        a = a.reshape(int(len(a) / 97), 97)
        df = dd.DataFrame(a)

        # Remove number before data
        for col in df:
            df[col] = df[col].str[3:]

        # Rename columns
        import numpy as np

        df.columns = np.arange(1, 98)

        col = {
            1: "rainfall_rate_32bit",
            2: "rainfall_accumulated_32bit",
            3: "weather_code_synop_4680",
            4: "weather_code_synop_4677",
            5: "weather_code_metar_4678",
            6: "weather_code_nws",
            7: "reflectivity_32bit",
            8: "mor_visibility",
            9: "sample_interval",
            10: "laser_amplitude",
            11: "number_particles",
            12: "sensor_temperature",
            13: "sensor_serial_number",
            14: "firmware_iop",
            15: "firmware_dsp",
            16: "sensor_heating_current",
            17: "sensor_battery_voltage",
            18: "sensor_status",
            19: "start_time",
            20: "sensor_time",
            21: "sensor_date",
            22: "station_name",
            23: "station_number",
            24: "rainfall_amount_absolute_32bit",
            25: "error_code",
            26: "sensor_temperature_pcb",
            27: "sensor_temperature_receiver",
            28: "sensor_temperature_trasmitter",
            30: "rainfall_rate_16_bit_30",
            31: "rainfall_rate_16_bit_1200",
            32: "rainfall_accumulated_16bit",
            33: "reflectivity_16bit",
            34: "rain_kinetic_energy",
            35: "snowfall_rate",
            60: "number_particles_all",
            61: "list_particles",
            90: "raw_drop_concentration",
            91: "raw_drop_average_velocity",
            92: "raw_drop_number",
        }

        df = df.rename(col, axis=1)

        # Cast time
        df["time"] = dd.to_datetime(
            df["sensor_date"] + "-" + df["sensor_time"], format="%d.%m.%Y-%H:%M:%S"
        )
        df = df.drop(columns=["sensor_date", "sensor_time"])

        # Drop useless columns
        df.replace("", np.nan, inplace=True)
        df.dropna(how="all", axis=1, inplace=True)
        col_to_drop = [40, 41, 50, 51, 93, 94, 95, 96, 97]
        df = df.drop(columns=col_to_drop)

        # Trim weather_code_metar_4678 and weather_code_nws
        df["weather_code_metar_4678"] = df["weather_code_metar_4678"].str.strip()
        df["weather_code_nws"] = df["weather_code_nws"].str.strip()

        # Delete invalid columsn by check_L0A
        col_to_drop = [
            "sensor_temperature_trasmitter",
            "sensor_temperature_pcb",
            "rainfall_rate_16_bit_1200",
            "sensor_temperature_receiver",
            "snowfall_rate",
            "rain_kinetic_energy",
            "rainfall_rate_16_bit_30",
        ]
        df = df.drop(columns=col_to_drop)

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    files_glob_pattern = "*"  # There is only one file without extension

    ####----------------------------------------------------------------------.
    #### - Create L0 products
    run_L0(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        keep_l0a=keep_l0a,
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        lazy=lazy,
        single_netcdf=single_netcdf,
        # Custom arguments of the reader
        files_glob_pattern=files_glob_pattern,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
    )
