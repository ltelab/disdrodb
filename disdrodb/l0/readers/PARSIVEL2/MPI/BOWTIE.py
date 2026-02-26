#!/usr/bin/env python3
# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
import os

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file

TRACKS_DICT = {
    "M203": ("2024-08-16 02:53:21", "2024-09-24 07:59:57"),
    "M204": ("2024-09-27 08:00:00", "2024-10-20 07:59:57"),
    "M205": ("2024-10-23 08:00:01", "2024-11-28 13:00:01"),
    "M206": ("2024-12-01 08:00:02", "2024-12-30 07:59:57"),
    "M207": ("2025-01-04 08:00:01", "2025-02-11 10:25:15"),
}


def get_track_for_dataframe(df):
    """Retrieve ship track identifier."""
    df_start, df_end = df["time"].min(), df["time"].max()

    overlaps = []
    for key, (start, end) in TRACKS_DICT.items():
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        # check if df range lies within track coverage
        if df_start <= end and df_end >= start:
            overlaps.append(key)
    return overlaps


def read_tracks_file(tracks_filepath):
    """Read GPS master track file."""
    df = pd.read_csv(
        tracks_filepath,
        names=["time", "latitude", "longitude", "flag"],
        dtype={"time": str, "latitude": float, "longitude": float, "flag": str},
        sep="\t",  # tab-separated
        skiprows=1,  # skip the weird first line
        engine="c",  # speed up reading
    )
    df["time"] = pd.to_datetime(df["time"])
    return df


def add_gps_coordinates(df, filepath):
    """Add GPS coordinates to dataframe."""
    # Retrieve useful tracks ids
    tracks_ids = get_track_for_dataframe(df)

    if len(tracks_ids) == 0:
        df["latitude"] = np.nan
        df["longitude"] = np.nan
        return df

    # Retrieve station base directory
    station_base_dir = os.path.join(os.path.sep, *filepath.split(os.path.sep)[:-1])
    # Define GPS files to read
    tracks_filepaths = [os.path.join(station_base_dir, f"{tracks_id}_mastertrack.zip") for tracks_id in tracks_ids]
    # Read GPS files
    list_df_tracks = [read_tracks_file(fpath) for fpath in tracks_filepaths]
    df_tracks = pd.concat(list_df_tracks)
    df_tracks = df_tracks.dropna(subset=["time"])

    # Ensure dataframes are sorted by time
    df = df.sort_values("time")
    df_tracks = df_tracks.sort_values("time")

    # Remove bad flags
    # df_tracks = df_tracks[df_tracks["flag"] == "1"]

    # Remove flag column
    df_tracks = df_tracks.drop(columns="flag")

    # Add GPS coordinate to dataframe
    df = pd.merge_asof(
        df,
        df_tracks,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("5min"),
    )
    return df


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = ["TO_BE_PARSED"]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = "/\n"

    # Skip first row as columns names
    reader_kwargs["header"] = None

    # Skip first 2 rows
    reader_kwargs["skiprows"] = 1

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
    reader_kwargs["na_values"] = ["na", "", "error", "NA"]

    # - Define encoding
    reader_kwargs["encoding"] = "latin1"

    ##------------------------------------------------------------------------.
    #### Read the data
    df = read_raw_text_file(
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        logger=logger,
    )

    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Raise error if empty file
    if len(df) == 0:
        raise ValueError(f"{filepath} is empty.")

    # Select only rows with expected number of delimiters
    df = df[df["TO_BE_PARSED"].str.count(";") == 1107]

    # Raise error if no data left
    if len(df) == 0:
        raise ValueError(f"No valid data in {filepath}.")

    # Split by ; delimiter
    df = df["TO_BE_PARSED"].str.split(";", expand=True, n=19)

    # Assign column names
    names = [
        "date",
        "time",
        "rainfall_rate_32bit",
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        # "weather_code_synop_4677",
        # "weather_code_metar_4678",
        "reflectivity_32bit",
        "mor_visibility",
        "sample_interval",
        "laser_amplitude",
        "number_particles",
        "sensor_temperature",
        "sensor_serial_number",
        "firmware_iop",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "sensor_status",
        "station_name",
        "rainfall_amount_absolute_32bit",
        "error_code",
        "ARRAY_TO_SPLIT",
    ]

    df.columns = names

    # Define time in datetime format
    time_str = df["date"] + " " + df["time"]
    df["time"] = pd.to_datetime(time_str, format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["time"])

    # Add raw array
    df["raw_drop_concentration"] = df["ARRAY_TO_SPLIT"].str[:224]
    df["raw_drop_average_velocity"] = df["ARRAY_TO_SPLIT"].str[224:448]
    df["raw_drop_number"] = df["ARRAY_TO_SPLIT"].str[448:]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "date",
        "station_name",
        "firmware_iop",
        "ARRAY_TO_SPLIT",
        "sensor_serial_number",
        "sample_interval",
    ]
    df = df.drop(columns=columns_to_drop)

    # Add GPS coordinates
    df = add_gps_coordinates(df, filepath=filepath)
    return df
