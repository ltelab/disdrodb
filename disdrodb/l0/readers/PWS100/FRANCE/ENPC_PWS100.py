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
"""DISDRODB reader for ENPC PWS100 raw text data."""
import zipfile

import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.utils.logger import log_error, log_warning


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""

    ##------------------------------------------------------------------------.
    #### Define function to read each txt file inside each daily zip file
    def read_txt_file(file, filename, logger):  # noqa PLR0911
        """Parse a single txt file within the daily zip file."""
        # Read file
        try:
            txt = file.readline().decode("utf-8")
        except Exception:
            log_warning(logger=logger, msg=f"{filename} is corrupted", verbose=False)
            return None

        # Check file is not empty
        if txt == "":
            log_warning(logger=logger, msg=f"{filename} is empty", verbose=False)
            return None

        if "PSU voltage too low" in txt or "volt" in txt:
            log_warning(logger=logger, msg=f"PSU voltage too low in {filename}", verbose=False)
            return None

        if "Error - message" in txt:
            log_warning(logger=logger, msg=f"Error message in {filename}", verbose=False)
            return None

        # Clean up the line
        txt = txt.replace(" 00 ", " 0 0 ")
        txt = txt.replace("  ", " 0 ")
        txt = txt[1:-8]

        # Split the cleaned line
        buf = txt.split(" ")

        # Helper to convert list of floats to comma-separated string
        def int_list_to_str(lst):
            return ",".join(f"{int(i)}" for i in lst)

        # Try to get the drop_size distribution:
        try:
            drop_size_distribution = int_list_to_str(buf[30:330])  # Drop size distribution (message field 42)
        except Exception:
            log_warning(logger, msg=f"Corrupted drop_size_distribution field in {filename}", verbose=False)
            return None

        # Try to get peak_to_pedestal_hist
        try:
            peak_to_pedestal_hist = int_list_to_str(buf[1499:1549])
        except Exception:
            log_warning(
                logger,
                msg=f"Corrupted raw_drop_number or peak_to_pedestal_hist field in {filename}",
                verbose=False,
            )
            return None
        # Parse fields
        data = {
            "mor_visibility": float(buf[2]),  # Visibility Range (message field 20)
            "weather_code_synop_4680": float(buf[3]),  # Present Weather Code (WMO)  (message field 21)
            "weather_code_metar_4678": buf[4],  # Present Weather Code (METAR)  (message field 22)
            "weather_code_nws": buf[5],  # Present Weather Code (NWS)  (message field 23)
            "alarms": int_list_to_str(buf[6:22]),  # Alarms (message field (24))
            "sensor_status": buf[22],  # Fault status of PWS100 (message field 25)
            "air_temperature": float(buf[23]),  # Temperature (°C) (message field 30)
            "relative_humidity": float(buf[24]),  # Sampled relative humidity (%) (message field 30)
            "wetbulb_temperature": float(buf[25]),  # Average wetbulb temperature (°C)(message field 30)
            "air_temperature_max": float(buf[26]),  # Maximum temperature (°C)(message field 31)
            "air_temperature_min": float(buf[27]),  # Minimum temperature (°C)(message field 31)
            "rainfall_rate": float(buf[28]),  # Precipitation rate (mm/h)(message field 40)
            "rainfall_accumulated": float(buf[29]),  # Precipitation accumulation (mm/h)(message field 41)
            "drop_size_distribution": drop_size_distribution,  # Drop size distribution (message field 42)
            "average_drop_velocity": float(buf[330]),  # Average velocity (mm/s)(message field 43)
            "average_drop_size": float(buf[331]),  # Average size (mm/h)(message field 43)
            "type_distribution": int_list_to_str(buf[332:343]),  # Type distribution (message field 44)
            "raw_drop_number": int_list_to_str(buf[343:1499]),  # Size/velocity spectrum (34*34) (message field 47)
            "peak_to_pedestal_hist": (
                peak_to_pedestal_hist  # Peak to pedestal ratio distribution histogram (message field 48)
            ),
        }

        # Convert to single-row DataFrame
        df = pd.DataFrame([data])

        # Define datetime "time" column from filename
        datetime_str = " ".join(filename.replace(".txt", "").split("_")[-6:])
        df["time"] = pd.to_datetime(datetime_str, format="%Y %m %d %H %M %S")

        # # Drop columns not agreeing with DISDRODB L0 standards
        # columns_to_drop = [
        #      "peak_to_pedestal_hist",
        #      "type_distribution",
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
                with zip_ref.open(filename) as f:
                    try:
                        df = read_txt_file(file=f, filename=filename, logger=logger)
                        if df is not None:
                            list_df.append(df)
                    except Exception as e:
                        msg = f"An error occurred while reading {filename}. The error is: {e}."
                        log_error(logger=logger, msg=msg, verbose=True)

    # Concatenate all dataframes into a single one
    df = pd.concat(list_df)

    # ---------------------------------------------------------------------.
    return df
