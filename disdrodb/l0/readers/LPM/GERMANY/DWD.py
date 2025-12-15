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
import os

import numpy as np
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file

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
    "number_particles_unknown_classification_internal_data",
    "number_particles_class_1",
    "number_particles_class_1_internal_data",
    "number_particles_class_2",
    "number_particles_class_2_internal_data",
    "number_particles_class_3",
    "number_particles_class_3_internal_data",
    "number_particles_class_4",
    "number_particles_class_4_internal_data",
    "number_particles_class_5",
    "number_particles_class_5_internal_data",
    "number_particles_class_6",
    "number_particles_class_6_internal_data",
    "number_particles_class_7",
    "number_particles_class_7_internal_data",
    "number_particles_class_8",
    "number_particles_class_8_internal_data",
    "number_particles_class_9",
    "number_particles_class_9_internal_data",
    "raw_drop_number",
]


def read_synop_file(filepath, logger):
    """Read SYNOP 10 min file."""
    ##------------------------------------------------------------------------.
    #### Define column names
    column_names = [
        "time",
        "temperature_2m",
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
        "number_particles_unknown_classification_internal_data",
        "number_particles_class_1",
        "number_particles_class_1_internal_data",
        "number_particles_class_2",
        "number_particles_class_2_internal_data",
        "number_particles_class_3",
        "number_particles_class_3_internal_data",
        "number_particles_class_4",
        "number_particles_class_4_internal_data",
        "number_particles_class_5",
        "number_particles_class_5_internal_data",
        "number_particles_class_6",
        "number_particles_class_6_internal_data",
        "number_particles_class_7",
        "number_particles_class_7_internal_data",
        "number_particles_class_8",
        "number_particles_class_8_internal_data",
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
        # "number_particles_unknown_classification_internal_data",
        # "number_particles_class_1",
        # "number_particles_class_1_internal_data",
        # "number_particles_class_2",
        # "number_particles_class_2_internal_data",
        # "number_particles_class_3",
        # "number_particles_class_3_internal_data",
        # "number_particles_class_4",
        # "number_particles_class_4_internal_data",
        # "number_particles_class_5",
        # "number_particles_class_5_internal_data",
        # "number_particles_class_6",
        # "number_particles_class_6_internal_data",
        # "number_particles_class_7",
        # "number_particles_class_7_internal_data",
        # "number_particles_class_8",
        # "number_particles_class_8_internal_data",
        # "number_particles_class_9",
        # "number_particles_class_9_internal_data",
        "raw_drop_number",
    ]
    df.columns = column_names

    # Sanitize columns
    df["current_heating_voltage_supply"] = df["current_heating_voltage_supply"].str.replace("///", "NaN")
    df["current_heating_house"] = df["current_heating_house"].str.replace("////", "NaN")
    df["current_heating_heads"] = df["current_heating_heads"].str.replace("////", "NaN")
    df["current_heating_carriers"] = df["current_heating_carriers"].str.replace("////", "NaN")

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
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        logger=logger,
    )
    ##------------------------------------------------------------------------.
    #### Adapt the dataframe to adhere to DISDRODB L0 standards
    filename = os.path.basename(filepath)
    if filename.startswith("3_"):
        return parse_format_v3(df)
    if filename.startswith("2_"):
        return parse_format_v2(df)
    if filename.startswith("1_"):
        return parse_format_v1(df)
    raise ValueError(f"Not implemented parser for DWD {filepath} data format.")
