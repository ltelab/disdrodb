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
"""DISDRODB reader for GID LPM sensors not measuring wind."""
import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file


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
    # Count number of delimiters to identify valid rows
    df = df[df["TO_PARSE"].str.count(";") == 521]

    # Split by ; delimiter (before raw drop number)
    df = df["TO_PARSE"].str.split(";", expand=True, n=80)

    # Assign column names
    column_names = [
        "time",
        "start_identifier",
        "device_address",
        "sensor_serial_number",
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
        "static_signal",
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
        "reserve_status",
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

    # Remove checksum from raw_drop_number
    df["raw_drop_number"] = df["raw_drop_number"].str.rsplit(";", n=2, expand=True)[0]

    # Define datetime "time" column
    time = df["time"].str.extract(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})")[0]
    df["time"] = pd.to_datetime(time, format="%d/%m/%Y %H:%M:%S", errors="coerce")

    # Drop row if start_identifier different than 00
    df["start_identifier"] = df["start_identifier"].astype(str).str[-2:]
    df = df[df["start_identifier"] == "00"]

    # Drop rows with invalid raw_drop_number
    df = df[df["raw_drop_number"].astype(str).str.len() == 1759]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "start_identifier",
        "device_address",
        "sensor_serial_number",
        "sensor_date",
        "sensor_time",
    ]
    df = df.drop(columns=columns_to_drop)
    return df
