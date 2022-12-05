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

    ##------------------------------------------------------------------------.
    #### - Define column names
    column_names = ["temp"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
    # - Skip first row as columns names
    reader_kwargs["header"] = None
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

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    def df_sanitizer_fun(df, lazy=False):
        # - Import dask or pandas
        import numpy as np

        if lazy:
            import pandas as dd

            df = df.compute()
        else:
            import pandas as dd

        # - Reshape dataframe
        arr = df.to_numpy()
        arr = arr.reshape(int(len(arr) / 97), 97)
        df = dd.DataFrame(arr)

        # - Remove number before data
        for col in df:
            df[col] = df[col].str[3:]

        # - Assign column names
        df.columns = np.arange(1, 98)
        valid_column_dict = {
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
            30: "rainfall_rate_16_bit_30",
            31: "rainfall_rate_16_bit_1200",
            32: "rainfall_accumulated_16bit",
            90: "raw_drop_concentration",
            91: "raw_drop_average_velocity",
            92: "raw_drop_number",
        }
        df = df.rename(valid_column_dict, axis=1)

        # - Keep only valid columns
        df = df[list(valid_column_dict.values())]

        # - Define datetime "time" column
        df["time"] = dd.to_datetime(
            df["sensor_date"] + "-" + df["sensor_time"], format="%d.%m.%Y-%H:%M:%S"
        )

        # - Trim weather_code_metar_4678 and weather_code_nws columns
        df["weather_code_metar_4678"] = df["weather_code_metar_4678"].str.strip()
        df["weather_code_nws"] = df["weather_code_nws"].str.strip()

        # - Drop columns not agreeing with DISDRODB L0 standards
        columns_to_drop = [
            "sensor_date",
            "sensor_time",
            # "rainfall_rate_16_bit_1200", "rainfall_rate_16_bit_30"
        ]  # TODO: check if metadata is OTT_Parsivel2
        df = df.drop(columns=columns_to_drop)

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_id>
    files_glob_pattern = "*.dat"

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
