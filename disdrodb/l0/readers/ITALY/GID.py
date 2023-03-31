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
"""Reader for the GID LPM network."""
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
    column_names = ["TO_BE_SPLITTED"]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\n"
    # Skip first row as columns names
    reader_kwargs["header"] = None
    # - Avoid first column to become df index
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
    reader_kwargs["na_values"] = ["na", "", "error", "NA", "-.-"]

    ##------------------------------------------------------------------------.
    #### - Define dataframe sanitizer function for L0 processing
    def df_sanitizer_fun(df):
        # - Import pandas
        import pandas as pd

        # - Split columns
        df = df["TO_BE_SPLITTED"].str.split(";", n=79, expand=True)

        # - Assign column names
        column_names = [
            "start_identifier",
            "sensor_serial_number",
            "software_version",
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

        # - Remove checksum at end of raw_drop_number
        df["raw_drop_number"] = df["raw_drop_number"].str.slice(stop=1760)

        # - Define 'time column
        df["time"] = df["sensor_date"].astype(str) + " " + df["sensor_time"].astype(str)

        # - Convert time column to datetime
        df["time"] = pd.to_datetime(df["time"], format="%d.%m.%y %H:%M:%S", errors="coerce")

        # - Drop columns not agreeing with DISDRODB L0 standards
        columns_to_drop = [
            "start_identifier",
            "software_version",
            "sensor_serial_number",
            "sensor_date",
            "sensor_time",
        ]
        df = df.drop(columns=columns_to_drop)

        df = df.drop(columns=["sensor_date", "sensor_time"])

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in <raw_dir>/data/<station_name>
    glob_patterns = "*.txt*"

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
