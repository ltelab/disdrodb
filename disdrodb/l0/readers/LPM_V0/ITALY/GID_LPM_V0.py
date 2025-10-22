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
"""DISDRODB reader for GID LPM V0 sensor (TC-TO) with incorrect reported time."""
import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file
from disdrodb.utils.logger import log_error, log_warning


def read_txt_file(file, filename, logger):
    """Parse for TC-TO LPM hourly file."""
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
    # Raise error if empty file
    if len(df) == 0:
        raise ValueError(f"{filename} is empty.")
        
    # Select only rows with expected number of delimiters 
    df = df[df["TO_PARSE"].str.count(";") == 442]

    # Check there are still valid rows
    if len(df) == 0:
        raise ValueError(f"No valid rows in {filename}.")

    # Split by ; delimiter (before raw drop number)
    df = df["TO_PARSE"].str.split(";", expand=True, n=43)

    # Assign column names
    names = [
        "start_identifier",
        "sensor_serial_number",
        "weather_code_synop_4680_5min",
        "weather_code_metar_4678_5min",
        "precipitation_rate_5min",
        "weather_code_synop_4680",
        "weather_code_metar_4678",
        "precipitation_rate",
        "precipitation_accumulated",
        "sensor_time",
        "temperature_interior",
        "laser_temperature",
        "laser_current_average",
        "control_voltage",
        "optical_control_voltage_output",
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
        "total_gross_volume_unknown_classification",
        "number_particles_hail",
        "total_gross_volume_hail",
        "number_particles_solid_precipitation",
        "total_gross_volume_solid_precipitation",
        "number_particles_great_pellet",
        "total_gross_volume_great_pellet",
        "number_particles_small_pellet",
        "total_gross_volume_small_pellet",
        "number_particles_snowgrain",
        "total_gross_volume_snowgrain",
        "number_particles_rain",
        "total_gross_volume_rain",
        "number_particles_small_rain",
        "total_gross_volume_small_rain",
        "number_particles_drizzle",
        "total_gross_volume_drizzle",
        "raw_drop_number",
    ]
    df.columns = names
    
    # Deal with case if there are 61 timesteps
    # - Occurs sometimes when previous hourly file miss timesteps
    if len(df) == 61:
        log_warning(logger=logger, msg=f"{filename} contains 61 timesteps. Dropping the first.")
        df = df.iloc[1:]

    # Raise error if more than 60 timesteps/rows
    n_rows = len(df)
    if n_rows > 60:
        raise ValueError(f"The hourly file contains {n_rows} timesteps.")

    # Infer and define "time" column
    start_time_str = filename.split(".")[0]  # '2024020200.txt'
    start_time = pd.to_datetime(start_time_str, format="%Y%m%d%H")

    # - Define timedelta based on sensor_time
    dt = pd.to_timedelta(df["sensor_time"] + ":00").to_numpy().astype("m8[s]")
    rollover_indices = np.where(np.diff(dt) < np.timedelta64(0, 's'))[0]
    if rollover_indices.size > 0:
        for idx in rollover_indices:
            dt[idx + 1:] += np.timedelta64(24, 'h')
    dt = dt - dt[0]

    # - Define approximate time
    df["time"] = start_time + dt

    # - Keep rows where time increment is between 00 and 59 minutes
    valid_rows = dt <= np.timedelta64(3540, "s")
    df = df[valid_rows]

    # Drop rows with invalid raw_drop_number
    # --> 440 value # 22x20
    df = df[df["raw_drop_number"].astype(str).str.len() == 1599]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "sensor_time",
        "start_identifier",
        "sensor_serial_number",
    ]
    df = df.drop(columns=columns_to_drop)
    return df


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    import zipfile

    ##------------------------------------------------------------------------.
    # filename = os.path.basename(filepath)
    # return read_txt_file(file=filepath, filename=filename, logger=logger)

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
            if filename.endswith(".txt"):
                # Open file
                with zip_ref.open(filename) as file:
                    try:
                        df = read_txt_file(file=file, filename=filename, logger=logger)
                        if df is not None:
                            list_df.append(df)
                    except Exception as e:
                        msg = f"An error occurred while reading {filename}. The error is: {e}"
                        log_error(logger=logger, msg=msg, verbose=True)

    # Check the zip file contains at least some non.empty files
    if len(list_df) == 0:
        raise ValueError(f"{filepath} contains only empty files!")

    # Concatenate all dataframes into a single one
    df = pd.concat(list_df)

    # ---------------------------------------------------------------------.
    return df
