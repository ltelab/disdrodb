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

    ####----------------------------------------------------------------------.
    ###########################
    #### CUSTOMIZABLE CODE ####
    ###########################
    #### - Define raw data headers
    column_names = ["TEMPORARY"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\n"

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"

    # Skip the first row (header)
    reader_kwargs["skiprows"] = 0

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
    def df_sanitizer_fun(df, lazy=False):
        import numpy as np

        # Import dask or pandas
        if lazy:
            import pandas as dd

            df = df.compute()
        else:
            import pandas as dd

        # Reshape dataframe and define dummy column names
        # - Assume always 97 fields
        n_fields = 97
        arr = df.to_numpy()
        arr = arr.reshape(int(len(arr) / n_fields), n_fields)
        df = dd.DataFrame(arr)
        df.columns = np.arange(1, n_fields + 1)

        # Define known field names
        column_names_dict = {
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
            # 26: 'sensor_temperature_pcb',
            # 28: 'sensor_temperature_receiver',
            # 29: 'sensor_temperature_trasmitter',
            30: "rainfall_rate_16_bit_30",
            31: "rainfall_rate_16_bit_1200",
            32: "rainfall_accumulated_16bit",
            # 33: 'reflectivity_16bit',
            # 34: 'rain_kinetic_energy',
            # 35: 'snowfall_rate',
            # 60: 'number_particles_all',
            # 61: 'list_particles',
            90: "raw_drop_concentration",
            91: "raw_drop_average_velocity",
            92: "raw_drop_number",
        }

        # Rename columns and drop unknown field names
        column_names = list(column_names_dict.values())
        df = df.rename(column_names_dict, axis=1)
        df = df[column_names]

        # Drop columns not meeting DISDRODB standard
        col_to_drop = [
            "sensor_serial_number",
            "firmware_iop",
            "firmware_dsp",
            "station_name",
            "station_number",
            "sample_interval",
        ]
        df = df.drop(columns=col_to_drop)

        # Remove field number. Example: '<field_number>: 24.8' becomes '24.8'
        for col in df:
            df[col] = df[col].str[3:]

        # Drop columns where values have been not logged
        # - This varies between stations
        # - We assume that the logging does not change across files !!!
        df = df.replace("", np.nan)
        df = df.dropna(how="all", axis=1)

        # Define time
        df["time"] = dd.to_datetime(
            df["sensor_date"] + "-" + df["sensor_time"], format="%d.%m.%Y-%H:%M:%S"
        )
        df = df.drop(columns=["sensor_date", "sensor_time", "start_time"])

        # Reformat weather codes
        if "weather_code_metar_4678" in df.columns:
            df["weather_code_metar_4678"] = df["weather_code_metar_4678"].str.strip()
        if "weather_code_nws" in df.columns:
            df["weather_code_nws"] = df["weather_code_nws"].str.strip()

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_id>
    files_glob_pattern = "*.dat"  # There is only one file without extension

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
        lazy=False,
        single_netcdf=single_netcdf,
        # Custom arguments of the reader
        files_glob_pattern=files_glob_pattern,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
    )
