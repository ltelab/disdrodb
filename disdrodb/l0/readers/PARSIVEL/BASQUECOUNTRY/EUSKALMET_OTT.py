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
"""DISDRODB reader for EUSKALMET OTT Parsivel raw data."""
# import os
# import tempfile
# from disdrodb.utils.compression import unzip_file_on_terminal

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file
from disdrodb.utils.logger import log_error

COLUMN_DICT = {
    "01": "rainfall_rate_32bit",
    "02": "rainfall_accumulated_32bit",
    "03": "weather_code_synop_4680",
    "04": "weather_code_synop_4677",
    "05": "weather_code_metar_4678",  # empty
    "06": "weather_code_nws",  # empty
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
    "30": "rainfall_rate_16bit",
    "31": "rainfall_rate_12bit",
    "32": "rainfall_accumulated_16bit",
    "90": "raw_drop_concentration",
    "91": "raw_drop_average_velocity",
    "93": "raw_drop_number",
}


def infill_missing_columns(df):
    """Infill with NaN missing columns."""
    columns = set(COLUMN_DICT.values())
    for c in columns:
        if c not in df.columns:
            df[c] = "NaN"
    return df


def read_txt_file(file, filename, logger):
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

    ##--------------------------------\----------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Empty file, raise error
    if len(df) == 0:
        raise ValueError(f"{filename} is empty.")

    # Select rows with valid spectrum
    # df = df[df["TO_PARSE"].str.count(";") == 1191]  # 1112

    # Raise errof if corrupted file
    if len(df) == 4:
        raise ValueError(f"{filename} is corrupted.")

    # Extract string
    string = df["TO_PARSE"].iloc[4]

    # Split into lines
    decoded_text = string.encode().decode("unicode_escape")
    decoded_text = decoded_text.replace("'", "").replace('"', "")
    lines = decoded_text.split()

    # Extract time
    time_str = lines[0].split(",")[1]

    # Split each line at the first colon
    data = [line.split(":", 1) for line in lines if ":" in line]

    # Create the DataFrame
    df = pd.DataFrame(data, columns=["ID", "Value"])

    # Drop rows with invalid IDs
    valid_id_str = np.char.rjust(np.arange(0, 94).astype(str), width=2, fillchar="0")
    df = df[df["ID"].astype(str).isin(valid_id_str)]

    # Select only rows with values
    df = df[df["Value"].apply(lambda x: x is not None)]

    # Reshape dataframe
    df = df.set_index("ID").T

    # Assign column names
    df = df.rename(COLUMN_DICT, axis=1)

    # Keep only columns defined in the dictionary
    df = df.filter(items=list(COLUMN_DICT.values()))

    # Infill missing columns
    df = infill_missing_columns(df)

    # Add time column ad datetime dtype
    df["time"] = pd.to_datetime(time_str, format="%Y%m%d%H%M%S", errors="coerce")

    # Preprocess the raw spectrum and raw_drop_average_velocity
    # - Add 0 before every ; if ; not preceded by a digit
    # - Example: ';;1;;' --> '0;0;1;0;'
    df["raw_drop_number"] = df["raw_drop_number"].str.replace(r"(?<!\d);", "0;", regex=True)
    df["raw_drop_average_velocity"] = df["raw_drop_average_velocity"].str.replace(r"(?<!\d);", "0;", regex=True)

    # Return the dataframe adhering to DISDRODB L0 standards
    return df


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    import zipfile

    # ---------------------------------------------------------------------.
    #### Iterate over all files (aka timesteps) in the daily zip archive
    # - Each file contain a single timestep !
    # list_df = []
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     # Extract all files
    #     unzip_file_on_terminal(filepath, temp_dir)

    #     # Walk through extracted files
    #     for root, _, files in os.walk(temp_dir):
    #         for filename in sorted(files):
    #             if filename.endswith(".txt"):
    #                 full_path = os.path.join(root, filename)
    #                 try:
    #                     df = read_txt_file(file=full_path, filename=filename, logger=logger)
    #                     if df is not None:
    #                         list_df.append(df)
    #                 except Exception as e:
    #                     msg = f"An error occurred while reading {filename}: {e}"
    #                     log_error(logger=logger, msg=msg, verbose=True)

    list_df = []
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        filenames = sorted(zip_ref.namelist())
        for filename in filenames:
            if filename.endswith(".dat"):
                # Open file
                with zip_ref.open(filename) as file:
                    try:
                        df = read_txt_file(file=file, filename=filename, logger=logger)
                        if df is not None:
                            list_df.append(df)
                    except Exception as e:
                        msg = f"An error occurred while reading {filename}. The error is: {e}."
                        log_error(logger=logger, msg=msg, verbose=True)

    # Check the zip file contains at least some non.empty files
    if len(list_df) == 0:
        raise ValueError(f"{filepath} contains only empty files!")

    # Concatenate all dataframes into a single one
    df = pd.concat(list_df)

    # ---------------------------------------------------------------------.
    return df
