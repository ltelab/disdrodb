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
"""DISDRODB reader for ENPC PARSIVEL2 raw text data."""
import zipfile

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file
from disdrodb.utils.logger import log_error


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""

    ##------------------------------------------------------------------------.
    #### Define function to read each txt file inside each daily zip file
    def read_txt_file(file, filename):
        """Parse a single txt file within the daily zip file."""
        ##------------------------------------------------------------------------.
        #### Define column names
        column_names = ["TO_PARSE"]

        ##------------------------------------------------------------------------.
        #### Define reader options
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
        #   - Already included: '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
        #                       '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
        #                       'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
        reader_kwargs["na_values"] = ["na", "", "error"]

        ##------------------------------------------------------------------------.
        #### Read the data
        df = read_raw_text_file(
            filepath=file,
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
        df = df[df["Value"].apply(lambda x: x is not None)]

        # Drop rows with invalid IDs
        valid_id_str = np.char.rjust(np.arange(0, 94).astype(str), width=2, fillchar="0")
        df = df[df["ID"].astype(str).isin(valid_id_str)]

        # Create the dataframe with each row corresponding to a timestep
        # - Group rows based on when ID values restart
        groups = df.groupby((df["ID"].astype(int).diff() <= 0).cumsum())

        # Reshape the dataframe
        group_dfs = []
        for _, group in groups:
            group_df = group.set_index("ID").T
            group_dfs.append(group_df)

        # Merge each timestep dataframe
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
            # "20": "sensor_time",
            # "21": "sensor_date",
            # "22": "station_name",
            # "23": "station_number",
            "24": "rainfall_amount_absolute_32bit",
            "25": "error_code",
            "26": "sensor_temperature_pcb",
            "27": "sensor_temperature_receiver",
            "28": "sensor_temperature_trasmitter",
            "30": "rainfall_rate_16_bit_30",
            "31": "rainfall_rate_16_bit_1200",
            "32": "rainfall_accumulated_16bit",
            "34": "rain_kinetic_energy",
            "35": "snowfall_rate",
            "90": "raw_drop_concentration",
            "91": "raw_drop_average_velocity",
            "93": "raw_drop_number",
        }

        df = df.rename(column_dict, axis=1)

        # Keep only columns defined in the dictionary
        df = df[list(column_dict.values())]

        # Define datetime "time" column from filename
        datetime_str = " ".join(filename.replace(".txt", "").split("_")[-6:])
        df["time"] = pd.to_datetime(datetime_str, format="%Y %m %d %H %M %S")

        # # Drop columns not agreeing with DISDRODB L0 standards
        # columns_to_drop = [
        #     "sensor_date",
        #     "sensor_time",
        # "firmware_iop",
        # "firmware_dsp",
        # "sensor_serial_number",
        # "station_name",
        # "station_number",
        # ]
        # df = df.drop(columns=columns_to_drop)
        return df

    # ---------------------------------------------------------------------.
    #### Iterate over all files (aka timesteps) in the daily zip archive
    # - Each file contain a single timestep !
    list_df = []
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        filenames = sorted(zip_ref.namelist())
        for filename in filenames:
            if filename.endswith(".txt"):
                # Open file
                with zip_ref.open(filename) as file:
                    try:
                        df = read_txt_file(file=file, filename=filename)
                        list_df.append(df)
                    except Exception as e:
                        msg = f"An error occurred while reading {filename}. The error is: {e}."
                        log_error(logger=logger, msg=msg, verbose=True)

    # Concatenate all dataframes into a single one
    df = pd.concat(list_df)

    # ---------------------------------------------------------------------.
    return df
