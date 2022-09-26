#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:01:52 2022

@author: kimbo
"""
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

# -------------------------------------------------------------------------.
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
    # Notes
    # - In all files, the datalogger voltage hasn't the delimiter,
    #   so need to be split to obtain datalogger_voltage and rainfall_rate_32bit

    # "01","Rain intensity 32 bit",8,"mm/h","single_number"
    # "02","Rain amount accumulated 32 bit",7,"mm","single_number"
    # "03","Weather code SYNOP Table 4680",2,"","single_number"
    # "04","Weather code SYNOP Table 4677",2,"","single_number"
    # "05","Weather code METAR Table 4678",5,"","character_string"
    # "06","Weather code NWS",4,"","character_string"
    # "07","Radar reflectivity 32 bit",6,"dBZ","single_number"
    # "08","MOR visibility in precipitation",5,"m","single_number"
    # "09","Sample interval",5,"s","single_number"
    # "10","Signal amplitude of laser",5,"","single_number"
    # "11","Number of particles detected and validated",5,"","single_number"
    # "12","Temperature in sensor housing",3,"degree_Celsius","single_number"
    # "13","Sensor serial number",6,"","character_string"
    # "14","Firmware IOP",6,"","character_string"
    # "15","Firmware DSP",6,"","character_string"
    # "16","Sensor head heating current",4,"A","single_number"
    # "17","Power supply voltage",4,"V","single_number"
    # "18","Sensor status",1,"","single_number"
    # "19","Date/time measuring start",19,"DD.MM.YYYY_hh:mm:ss","character_string"
    # "20","Sensor time",8,"hh:mm:ss","character_string"
    # "21","Sensor date",10,"DD.MM.YYYY","character_string"
    # "22","Station name",4,"","character_string"
    # "23","Station number",4,"","character_string"
    # "24","Rain amount absolute 32 bit",7,"mm","single_number"
    # "25","Error code",3,"","character_string"
    # "26","Temperature PCB",3,"degree_Celsius","single_number"
    # "27","Temperature in right sensor head",3,"degree_Celsius","single_number"
    # "28","Temperature in left sensor head",3,"degree_Celsius","single_number"
    # "30","Rain intensity 16 bit max 30 mm/h",6,"mm/h","single_number"
    # "31","Rain intensity 16 bit max 1200 mm/h",6,"mm/h","single_number"
    # "32","Rain amount accumulated 16 bit",7,"mm","single_number"
    # "33","Radar reflectivity 16 bit",5,"dBZ","single_number"
    # "34","Kinetic energy",7,"J/(m2*h)","single_number"
    # "35","Snowfall intensity",7,"mm/h","single_number"
    # "60","Number of all particles detected",8,"","single_number"
    # "61","List of all particles detected",13,"","list"
    # "90","raw_drop_concentration",224,"","vector"
    # "91","raw_drop_average_velocity",224,"","vector"
    # "93","Raw data",4096,"","matrix"

    column_names = ["time", "epoch_time", "TO_BE_PARSED"]

    ##------------------------------------------------------------------------.
    #### - Define reader options

    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ";"

    # - Avoid first column to become df index !!!
    reader_kwargs["index_col"] = False

    # - Define behaviour when encountering bad lines
    reader_kwargs["on_bad_lines"] = "skip"

    # - Define parser engine
    #   - C engine is faster
    #   - Python engine is more feature-complete
    reader_kwargs["engine"] = "python"

    # - Define on-the-fly decompression of on-disk data
    #   - Available: gzip, bz2, zip
    reader_kwargs["compression"] = "infer"
    # reader_kwargs['zipped'] = False
    # reader_kwargs['zipped'] = True
    # - Strings to recognize as NA/NaN and replace with standard NA flags
    #   - Already included: ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’,
    #                       ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’,
    #                       ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
    reader_kwargs["na_values"] = [
        "na",
        "",
        "error",
        "NA",
        "-.-",
        " NA",
    ]

    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    # Cast all to string
    reader_kwargs["dtype"] = str
    # reader_kwargs["file_name_to_read_zipped"] = 'tar'

    # Skip first row as columns names
    reader_kwargs["header"] = None

    ##------------------------------------------------------------------------.
    #### - Define facultative dataframe sanitizer function for L0 processing
    # - Enable to deal with bad raw data files
    # - Enable to standardize raw data files to L0 standards  (i.e. time to datetime)
    df_sanitizer_fun = None

    def df_sanitizer_fun(df, lazy=False):
        # Import dask or pandas
        if lazy:
            import dask.dataframe as dd
        else:
            import pandas as dd

        # Station 8 has all raw_drop_number corrupted, so it can't be use
        # Bug on rows 105000 and 110000 for station 7 (000NETDL07 and PAR007 device name) on dask
        if lazy:
            df = df.compute()

        # if (df['TO_BE_PARSED'].str.contains('000NETDL07')).any() | (df['TO_BE_PARSED'].str.contains('PAR007')).any():
        #     # df = df.loc[105000:110000]
        #     df.drop(df.index[105000:110000], axis=0, inplace=True)

        # # Remove rows that have a corrupted "TO_BE_PARSED" column
        # df = df.loc[df["TO_BE_PARSED"].astype(str).str.len() > 50]

        # Check date lenght
        df_date_with_error = df.loc[df["time"].astype(str).str.len() != 15]
        if len(df_date_with_error.index) > 0:
            raise Exception("Some dates do not have 15 characters")

        df = df.loc[df["time"].astype(str).str.len() == 15]

        df["time"] = dd.to_datetime(df["time"], format="%Y%m%d-%H%M%S")

        # Remove dupliacte time
        df = df.drop_duplicates(subset=["time"])

        # Extract the last column that contains the 37 remaining attributes
        df_concatenated_column = df["TO_BE_PARSED"].str.split(";", expand=True, n=99)

        # try:
        #     df['time'] = d
        # d.to_datetime(df['time'], format='%Y%m%d-%H%M%S')
        # except ValueError:
        #     df['time'] = dd.to_datetime(df['time'], format='%Y%m%d-%H%M%S', errors='coerce')
        #     df = df.loc[df.time.notnull()]

        # remove TO_BE_PARSED column
        df = df.drop(["TO_BE_PARSED"], axis=1)

        # Add names to columns
        column_names_concatenated_column = [
            "rainfall_rate_32bit",
            "rainfall_accumulated_32bit",
            "weather_code_synop_4680",
            "weather_code_synop_4677",
            "weather_code_metar_4678",
            "weather_code_nws",
            "reflectivity_32bit",
            "mor_visibility",
            "sample_interval",
            "laser_amplitude",
            "number_particles",
            "sensor_temperature",
            "sensor_serial_number",
            "firmware_iop",
            "firmware_dsp",
            "sensor_heating_current",
            "sensor_battery_voltage",
            "sensor_status",
            "date_time_measurement_start",
            "sensor_time",
            "sensor_date",
            "station_name",
            "station_number",
            "rainfall_amount_absolute_32bit",
            "error_code",
            "sensor_temperature_pcb",
            "sensor_temperature_receiver",
            "sensor_temperature_trasmitter",
            "rainfall_rate_16_bit_30",
            "rainfall_rate_16_bit_1200",
            "rainfall_accumulated_16bit",
            "reflectivity_16bit",
            "rain_kinetic_energy",
            "snowfall_rate",
            "number_particles_all",
            "number_particles_all_detected",
        ]

        # Remove unnecessary columns
        df_unconcatenated_column = df_concatenated_column.iloc[
            :, : len(column_names_concatenated_column)
        ]

        # Renamed columns
        df_unconcatenated_column = df_unconcatenated_column.set_axis(
            column_names_concatenated_column, axis=1, inplace=False
        )

        # Remove char from rain intensity
        df_unconcatenated_column["rainfall_rate_32bit"] = df_unconcatenated_column[
            "rainfall_rate_32bit"
        ].str.lstrip("b'")

        # Remove spaces on weather_code_metar_4678 and weather_code_nws
        df_unconcatenated_column["weather_code_metar_4678"] = df_unconcatenated_column[
            "weather_code_metar_4678"
        ].str.strip()
        df_unconcatenated_column["weather_code_nws"] = df_unconcatenated_column[
            "weather_code_nws"
        ].str.strip()

        # Add the comma on the raw_drop_concentration, raw_drop_average_velocity and raw_drop_number
        df_raw_drop_concentration = (
            df_concatenated_column.iloc[
                :, len(column_names_concatenated_column) - 1 : 67
            ]
            .apply(lambda x: ",".join(x.dropna().astype(str)), axis=1)
            .to_frame("raw_drop_concentration")
        )
        df_raw_drop_average_velocity = (
            df_concatenated_column.iloc[:, 67:-1]
            .apply(lambda x: ",".join(x.dropna().astype(str)), axis=1)
            .to_frame("raw_drop_average_velocity")
        )

        # Station 8 has \r\n' at end
        if (df_unconcatenated_column["station_name"] == "PAR008").any():
            df_raw_drop_number = (
                df_concatenated_column.iloc[:, -1:]
                .squeeze()
                .str.replace(r"(\w{3})", r"\1,", regex=True)
                .str.rstrip("\\r\\n'")
                .to_frame("raw_drop_number")
            )
        else:
            df_raw_drop_number = (
                df_concatenated_column.iloc[:, -1:]
                .squeeze()
                .str.replace(r"(\w{3})", r"\1,", regex=True)
                .str.rstrip("'")
                .to_frame("raw_drop_number")
            )

        # Concat all togheter
        df = dd.concat(
            [
                df,
                df_unconcatenated_column,
                df_raw_drop_concentration,
                df_raw_drop_average_velocity,
                df_raw_drop_number,
            ],
            axis=1,
        )

        # Drop invalid rows
        df = df.loc[df["raw_drop_concentration"].astype(str).str.len() == 223]
        df = df.loc[df["raw_drop_average_velocity"].astype(str).str.len() == 223]
        df = df.loc[df["raw_drop_number"].astype(str).str.len() == 4096]

        # Drop variables not required in L0 Apache Parquet
        todrop = [
            "firmware_iop",
            "firmware_dsp",
            "date_time_measurement_start",
            "sensor_time",
            "sensor_date",
            "station_name",
            "station_number",
            "sensor_serial_number",
            "epoch_time",
            "sample_interval",
            "sensor_serial_number",
            "number_particles_all_detected",
        ]

        df = df.drop(todrop, axis=1)

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    # files_glob_pattern= "*.tar.xz"
    files_glob_pattern = "*.csv"

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
        lazy=lazy,
        single_netcdf=single_netcdf,
        # Custom arguments of the parser
        files_glob_pattern=files_glob_pattern,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
    )
