#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:08:17 2023

@author: ghiggi
"""
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
    column_names = ["TO_PARSE"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"
    # - Skip first row as columns names
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
    #   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
    #                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
    #                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
    reader_kwargs["na_values"] = ["na", "", "error"]

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    def df_sanitizer_fun(df):
        # - Import pandas
        import pandas as pd

        # Create ID and Value columns
        df = df["TO_PARSE"].str.split(":", expand=True, n=1)
        df.columns = ["ID", "Value"]

        # Drop rows with no values
        df = df[df["Value"].astype(bool)]

        # Create the dataframe with each row corresponding to a timestep
        # - Group rows based on when ID values restart
        groups = df.groupby((df["ID"].astype(int).diff() <= 0).cumsum())

        # - Reshape the dataframe
        group_dfs = []
        for name, group in groups:
            group_df = group.set_index("ID").T
            group_dfs.append(group_df)

        # - Merge each timestep dataframe
        # --> Missing columns are infilled by NaN
        df = pd.concat(group_dfs, axis=0)

        # Assign column names
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
            # "19": "start_time",
            "20": "sensor_time",
            "21": "sensor_date",
            # "22": "station_name",
            # "23": "station_number",
            "24": "rainfall_amount_absolute_32bit",
            "25": "error_code",
            "30": "rainfall_rate_16_bit_30",
            "31": "rainfall_rate_16_bit_1200",
            "32": "rainfall_accumulated_16bit",
            "90": "raw_drop_concentration",
            "91": "raw_drop_average_velocity",
            "93": "raw_drop_number",
        }
        df = df.rename(column_dict, axis=1)

        # - Keep only columns defined in the dictionary
        df = df[list(column_dict.values())]

        # - Define datetime "time" column
        df["time"] = df["sensor_date"] + "-" + df["sensor_time"]
        df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y-%H:%M:%S", errors="coerce")

        # - Drop columns not agreeing with DISDRODB L0 standards
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

        # Preprocess the raw spectrum and raw_drop_average_velocity
        # - Add 0 before every ; if ; not preceded by a digit
        # - Example: ';;1;;' --> '0;0;1;0;'
        df["raw_drop_average_velocity"] = df["raw_drop_average_velocity"].str.replace(r"(?<!\d);", "0;", regex=True)
        df["raw_drop_number"] = df["raw_drop_number"].str.replace(r"(?<!\d);", "0;", regex=True)

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
