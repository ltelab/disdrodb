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
"""DISDRODB reader for DWD stations."""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file
from disdrodb.utils.logger import log_error, log_warning

# Assign column names
COLUMNS = [
    "weather_code_synop_4677_5min",
    "weather_code_synop_4680_5min",
    "weather_code_metar_4678_5min",
    "precipitation_rate_5min",
    "weather_code_synop_4677",
    "weather_code_synop_4680",
    "weather_code_metar_4678",
    "precipitation_rate",
    "rainfall_rate",
    "snowfall_rate",
    "precipitation_accumulated",
    "mor_visibility",
    "reflectivity",
    "quality_index",
    "max_hail_diameter",
    "laser_status",
    "static_signal_status",
    "laser_temperature_analog_status",
    "laser_temperature_digital_status",
    "laser_current_analog_status",
    "laser_current_digital_status",
    "sensor_voltage_supply_status",
    "current_heating_pane_transmitter_head_status",
    "current_heating_pane_receiver_head_status",
    "temperature_sensor_status",
    "current_heating_voltage_supply_status",
    "current_heating_house_status",
    "current_heating_heads_status",
    "current_heating_carriers_status",
    "control_output_laser_power_status",
    "reserved_status",
    "temperature_interior",
    "laser_temperature",
    "laser_current_average",
    "control_voltage",
    "optical_control_voltage_output",
    "sensor_voltage_supply",
    "current_heating_pane_transmitter_head",
    "current_heating_pane_receiver_head",
    "temperature_ambient",
    "current_heating_voltage_supply",
    "current_heating_house",
    "current_heating_heads",
    "current_heating_carriers",
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
    "number_particles_large_pellet",
    "total_gross_volume_large_pellet",
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
    "number_particles_class_9",
    "number_particles_class_9_internal_data",
    "raw_drop_number",
]

####------------------------------------------------------------------------.
#### SYNOP utilities


def _reindex_to_custom_frequency(df, freq="1min"):
    # Interpolate to 1 min
    t_start = df.index.min()
    t_end = df.index.max()
    timesteps = pd.date_range(
        start=t_start,
        end=t_end,
        freq=freq,
    )
    return df.reindex(timesteps)


def interpolate_wind_direction(wind_direction, limit=None):
    """Interpolate NaN values in a 1-min wind direction series.

    Use circular (vector) interpolation.

    Parameters
    ----------
    wind_direction : pandas.Series
        Wind direction in degrees with DateTimeIndex.
    limit : int or None
        Max number of consecutive NaNs to fill.

    Returns
    -------
    pandas.Series
        Wind direction with NaNs interpolated.
    """
    wind_direction = wind_direction.copy()
    wind_direction.index = pd.to_datetime(wind_direction.index)

    # Convert to radians
    theta = np.deg2rad(wind_direction)

    # Vector components
    u = np.cos(theta)
    v = np.sin(theta)

    df_vec = pd.DataFrame({"u": u, "v": v}, index=wind_direction.index)

    # Interpolate ONLY NaNs
    df_vec_i = df_vec.interpolate(
        method="time",
        limit=limit,
    )

    # Back to degrees
    dir_i = np.rad2deg(np.arctan2(df_vec_i["v"], df_vec_i["u"]))
    dir_i = (dir_i + 360) % 360

    dir_i = (dir_i / 10).round() * 10
    return pd.Series(dir_i, index=wind_direction.index, name=wind_direction.name)


def retrieve_synop_filepaths(df, filepath):
    """Retrieve SYNOP files relevant for a LPM file."""
    # Retrieve relevant info to list required synop files
    filename = os.path.basename(filepath)
    station_id = filename.split("_")[1]
    date = df["time"].dt.date.iloc[0]
    synop_base_dir = Path(filepath).parents[3] / "SYNOP"
    synop_filepaths = []
    for d in [date, date + pd.Timedelta(days=1)]:
        y = d.strftime("%Y")
        m = d.strftime("%m")
        ymd = d.strftime("%Y%m%d")
        fname_pattern = f"synop10min_{station_id}_{ymd}*1.0days.dat"
        glob_pattern = os.path.join(synop_base_dir, y, m, fname_pattern)
        files = glob.glob(glob_pattern)
        if len(files) >= 1:
            synop_filepaths.append(*files)
    return synop_filepaths


def read_synop_file(filepath, logger=None):
    """Read SYNOP 10 min file."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = [
        "time",
        "air_temperature",
        "relative_humidity",
        "precipitation_accumulated_10min",
        "total_cloud_cover",
        "wind_speed",
        "wind_direction",
    ]
    ##------------------------------------------------------------------------.
    #### Define reader options
    # - For more info: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    reader_kwargs = {}

    # - Define delimiter
    reader_kwargs["delimiter"] = r"\s+"

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # Since column names are expected to be passed explicitly, header is set to None
    reader_kwargs["header"] = None

    # - Number of rows to be skipped at the beginning of the file
    reader_kwargs["skiprows"] = 6

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
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        logger=logger,
    )

    # Define datetime "time" column
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S", errors="coerce")

    # Return SYNOP dataframe
    return df


def _add_nan_synop_variables(df, logger, msg):
    """Guarantee SYNOP columns on LPM df output."""
    # Define SYNOP vars to be always present
    synop_vars = [
        "air_temperature",
        "relative_humidity",
        "wind_speed",
        "wind_direction",
    ]
    # Add SYNOP vars columns
    log_warning(logger=logger, msg=msg)
    for v in synop_vars:
        df[v] = np.nan
    return df


def add_synop_information(df, filepath, logger):
    """Add SYNOP (10-min) meteorological data to an LPM (1-min) dataframe.

    LPM files contains timesteps: 00:00-23.59
    SYNOP files contains timesteps: 00:00-23.50

    To interpolate SYNOP data between 00:00-23.59 we need also next-day SYNOP file

    Always returns a dataframe with SYNOP columns present.
    """
    # Drop duplicate timesteps from input LPM dataframe
    df = df.drop_duplicates(subset="time", keep="first").sort_values("time")

    # Retrieve date
    date = df["time"].iloc[0].date()

    # --------------------------------------------------------------------
    # Retrieve required SYNOP files
    synop_filepaths = retrieve_synop_filepaths(df=df, filepath=filepath)

    # If no SYNOP files available
    if not synop_filepaths:
        msg = f"No SYNOP files available for {date}"
        return _add_nan_synop_variables(df, logger=logger, msg=msg)

    # Read relevant SYNOP files
    synop_dfs = []
    for f in synop_filepaths:
        try:
            synop_dfs.append(read_synop_file(f))
        except Exception as e:
            log_error(logger=logger, msg=f"Failed to read SYNOP file {f}. Error: {e!s}")

    if not synop_dfs:
        msg = f"No valid SYNOP data could be read for {date}"
        return _add_nan_synop_variables(df, logger=logger, msg=msg)

    # Concatenate SYNOP files into unique dataframe
    df_synop_10min = pd.concat(synop_dfs, ignore_index=True)

    # --------------------------------------------------------------------
    # Subset SYNOP file
    tmin = df["time"].min() - pd.Timedelta(minutes=10)
    tmax = df["time"].max() + pd.Timedelta(minutes=10)
    df_synop_10min = df_synop_10min[(df_synop_10min["time"] >= tmin) & (df_synop_10min["time"] <= tmax)]
    if df_synop_10min.empty:
        msg = f"No SYNOP data available for {date}"
        return _add_nan_synop_variables(df, logger=logger, msg=msg)

    # Drop time duplicates if present
    df_synop_10min = df_synop_10min.drop_duplicates(subset="time", keep="first")
    df_synop_10min = df_synop_10min.drop(columns=["total_cloud_cover", "precipitation_accumulated_10min"])
    # Reindex SYNOP 10 min file to 1 min
    df_synop_10min = df_synop_10min.set_index("time")  # set time column as index
    df_synop_10min = df_synop_10min.astype(float)  # cast column to float
    df_synop_1min = _reindex_to_custom_frequency(df_synop_10min, freq="1min")
    # Interpolate variables
    df_synop_1min["wind_direction"] = interpolate_wind_direction(df_synop_1min["wind_direction"], limit=9)
    variables = ["air_temperature", "relative_humidity", "wind_speed"]
    df_synop_1min[variables] = df_synop_1min[variables].interpolate(method="time", limit=9)
    df_synop_1min = df_synop_1min.reset_index().rename(columns={"index": "time"})
    # Merge data
    df_synop_1min = df_synop_1min.drop_duplicates(subset="time", keep="first").sort_values("time")
    df_merged = pd.merge_asof(
        df,
        df_synop_1min,
        on="time",
        direction="nearest",  # or "backward" / "forward"
        tolerance=pd.Timedelta("0min"),
    )
    return df_merged


####-------------------------------------------------------------------------.
#### LPM parsers


def parse_format_v1(df):
    """Parse DWD format v1."""
    raise NotImplementedError


def parse_format_v2(df):
    """Parse DWD format v2."""
    # Count number of delimiters to identify valid rows
    df = df[df["TO_PARSE"].str.count(";") == 520]

    # Split by ; delimiter (before raw drop number)
    df = df["TO_PARSE"].str.split(";", expand=True, n=81)

    # Assign column names
    column_names = [
        "dummy_1",
        "dummy_2",
        "start_identifier",
        "dummy_3",
        "device_address",
        "sensor_date",
        "sensor_time",
        "weather_code_synop_4677_5min",
        "weather_code_synop_4680_5min",
        "weather_code_metar_4678_5min",
        "precipitation_rate_5min",
        "weather_code_synop_4677",
        "weather_code_synop_4680",
        "weather_code_metar_4678",
        "precipitation_rate",
        "rainfall_rate",
        "snowfall_rate",
        "precipitation_accumulated",
        "mor_visibility",
        "reflectivity",
        "quality_index",
        "max_hail_diameter",
        "laser_status",
        "static_signal_status",
        "laser_temperature_analog_status",
        "laser_temperature_digital_status",
        "laser_current_analog_status",
        "laser_current_digital_status",
        "sensor_voltage_supply_status",
        "current_heating_pane_transmitter_head_status",
        "current_heating_pane_receiver_head_status",
        "temperature_sensor_status",
        "current_heating_voltage_supply_status",
        "current_heating_house_status",
        "current_heating_heads_status",
        "current_heating_carriers_status",
        "control_output_laser_power_status",
        "reserved_status",
        "temperature_interior",
        "laser_temperature",
        "laser_current_average",
        "control_voltage",
        "optical_control_voltage_output",
        "sensor_voltage_supply",
        "current_heating_pane_transmitter_head",
        "current_heating_pane_receiver_head",
        "temperature_ambient",
        "current_heating_voltage_supply",
        "current_heating_house",
        "current_heating_heads",
        "current_heating_carriers",
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
        "number_particles_large_pellet",
        "total_gross_volume_large_pellet",
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
        "number_particles_class_9",
        "number_particles_class_9_internal_data",
        "raw_drop_number",
    ]
    df.columns = column_names

    # Define datetime "time" column
    df["time"] = df["sensor_date"] + "-" + df["sensor_time"]
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%y-%H:%M:%S", errors="coerce")

    # Drop rows with invalid raw_drop_number
    df = df[df["raw_drop_number"].astype(str).str.len() == 1759]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "device_address",
        "start_identifier",
        "sensor_date",
        "sensor_time",
        "dummy_1",
        "dummy_2",
        "dummy_3",
    ]
    df = df.drop(columns=columns_to_drop)
    return df


def parse_format_v3(df):
    """Parse DWD format v3."""
    # Count number of delimiters to identify valid rows
    df = df[df["TO_PARSE"].str.count(";") == 498]

    # Split by ; delimiter (before raw drop number)
    df = df["TO_PARSE"].str.split(";", expand=True, n=59)

    # Assign column names
    column_names = [
        "dummy_1",
        "dummy_2",
        "dummy_3",
        "dummy_4",
        "device_address",
        "sensor_serial_number",
        "software_version",
        "dummy_5",
        "dummy_6",
        "sensor_date",
        "sensor_time",
        "weather_code_synop_4677_5min",
        "weather_code_synop_4680_5min",
        "weather_code_metar_4678_5min",
        "precipitation_rate_5min",
        "weather_code_synop_4677",
        "weather_code_synop_4680",
        "weather_code_metar_4678",
        "precipitation_rate",
        "rainfall_rate",
        "snowfall_rate",
        "precipitation_accumulated",
        "mor_visibility",
        "reflectivity",
        "quality_index",
        "max_hail_diameter",
        "laser_status",
        "static_signal_status",
        "laser_temperature_analog_status",
        "laser_temperature_digital_status",
        "laser_current_analog_status",
        "laser_current_digital_status",
        "sensor_voltage_supply_status",
        "current_heating_pane_transmitter_head_status",
        "current_heating_pane_receiver_head_status",
        "temperature_sensor_status",
        "current_heating_voltage_supply_status",
        "current_heating_house_status",
        "current_heating_heads_status",
        "current_heating_carriers_status",
        "control_output_laser_power_status",
        "reserved_status",
        "temperature_interior",
        "laser_temperature",
        "laser_current_average",
        "control_voltage",
        "optical_control_voltage_output",
        "sensor_voltage_supply",
        "current_heating_pane_transmitter_head",
        "current_heating_pane_receiver_head",
        "temperature_ambient",
        "current_heating_voltage_supply",
        "current_heating_house",
        "current_heating_heads",
        "current_heating_carriers",
        "number_particles",
        # "number_particles_internal_data",
        "number_particles_min_speed",
        # "number_particles_min_speed_internal_data",
        "number_particles_max_speed",
        # "number_particles_max_speed_internal_data",
        "number_particles_min_diameter",
        # "number_particles_min_diameter_internal_data",
        # "number_particles_no_hydrometeor",
        # "number_particles_no_hydrometeor_internal_data",
        # "number_particles_unknown_classification",
        # "total_gross_volume_unknown_classification",
        # "number_particles_hail",
        # "total_gross_volume_hail",
        # "number_particles_solid_precipitation",
        # "total_gross_volume_solid_precipitation",
        # "number_particles_large_pellet",
        # "total_gross_volume_large_pellet",
        # "number_particles_small_pellet",
        # "total_gross_volume_small_pellet",
        # "number_particles_snowgrain",
        # "total_gross_volume_snowgrain",
        # "number_particles_rain",
        # "total_gross_volume_rain",
        # "number_particles_small_rain",
        # "total_gross_volume_small_rain",
        # "number_particles_drizzle",
        # "total_gross_volume_drizzle",
        # "number_particles_class_9",
        # "number_particles_class_9_internal_data",
        "raw_drop_number",
    ]
    df.columns = column_names

    # Sanitize columns for //// and ??? patterns
    pattern = r"([/?]+)"
    for col in df.select_dtypes(include=["string", "object"]):
        df[col] = (
            df[col]
            .str.replace(pattern, "NaN", regex=True)
            .str.split(",")
            .apply(lambda x: ["NaN" if v.strip() == "NaN" else v.strip() for v in x])
            .str.join(", ")
        )

    # Define datetime "time" column
    df["time"] = df["sensor_date"] + "-" + df["sensor_time"]
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%y-%H:%M:%S", errors="coerce")

    # Drop rows with invalid raw_drop_number
    df = df[df["raw_drop_number"].astype(str).str.len() == 1759]

    # Identify missing columns and add NaN
    missing_columns = np.array(COLUMNS)[np.isin(COLUMNS, df.columns, invert=True)].tolist()
    if len(missing_columns) > 0:
        for column in missing_columns:
            df[column] = "NaN"

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "device_address",
        "sensor_serial_number",
        "software_version",
        "sensor_date",
        "sensor_time",
        "dummy_1",
        "dummy_2",
        "dummy_3",
        "dummy_4",
        "dummy_5",
        "dummy_6",
    ]
    df = df.drop(columns=columns_to_drop)
    return df


@is_documented_by(reader_generic_docstring)
def reader(
    filepath,
    logger=None,
):
    """Reader."""
    ##------------------------------------------------------------------------.
    #### Define raw data headers
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
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        logger=logger,
    )
    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    # Read LPM raw data
    filename = os.path.basename(filepath)
    if filename.startswith("3_"):
        df = parse_format_v3(df)
    elif filename.startswith("2_"):
        df = parse_format_v2(df)
    elif filename.startswith("1_"):
        df = parse_format_v1(df)
    else:
        raise ValueError(f"Not implemented parser for DWD {filepath} data format.")

    # Add SYNOP data if available
    df = add_synop_information(df=df, filepath=filepath, logger=logger)

    # Return dataframe
    return df
