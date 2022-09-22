#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:40:17 2022

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
import click
from disdrodb.L0 import run_L0

# -------------------------------------------------------------------------.
# CLIck Command Line Interface decorator
@click.command()  # options_metavar='<options>'
@click.argument("raw_dir", type=click.Path(exists=True), metavar="<raw_dir>")
@click.argument("processed_dir", metavar="<processed_dir>")
@click.option(
    "-l0a",
    "--l0a_processing",
    type=bool,
    show_default=True,
    default=True,
    help="Perform L0A processing",
)
@click.option(
    "-l0b",
    "--l0b_processing",
    type=bool,
    show_default=True,
    default=True,
    help="Perform L0B processing",
)
@click.option(
    "-k",
    "--keep_l0a",
    type=bool,
    show_default=True,
    default=True,
    help="Whether to keep the l0a Parquet file",
)
@click.option(
    "-f",
    "--force",
    type=bool,
    show_default=True,
    default=False,
    help="Force overwriting",
)
@click.option(
    "-v", "--verbose", type=bool, show_default=True, default=False, help="Verbose"
)
@click.option(
    "-d",
    "--debugging_mode",
    type=bool,
    show_default=True,
    default=False,
    help="Switch to debugging mode",
)
@click.option(
    "-l",
    "--lazy",
    type=bool,
    show_default=True,
    default=True,
    help="Use dask if lazy=True",
)
@click.option(
    "-s",
    "--single_netcdf",
    type=bool,
    show_default=True,
    default=True,
    help="Produce single netCDF",
)
def main(
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
    """Script to process raw data to L0A and L0B format.

    Parameters
    ----------
    raw_dir : str
        Directory path of raw file for a specific campaign.
        The path should end with <campaign_name>.
        Example raw_dir: '<...>/disdrodb/data/raw/<campaign_name>'.
        The directory must have the following structure:
        - /data/<station_id>/<raw_files>
        - /metadata/<station_id>.json
        Important points:
        - For each <station_id> there must be a corresponding JSON file in the metadata subfolder.
        - The <campaign_name> must semantically match between:
           - the raw_dir and processed_dir directory paths;
           - with the key 'campaign_name' within the metadata YAML files.
        - The campaign_name are set to be UPPER CASE.
    processed_dir : str
        Desired directory path for the processed L0A and L0B products.
        The path should end with <campaign_name> and match the end of raw_dir.
        Example: '<...>/disdrodb/data/processed/<campaign_name>'.
    L0A_processing : bool
      Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
      The default is True.
    L0B_processing : bool
      Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
      The default is True.
    keep_L0A : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default is False.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        - For L0A processing, it processes just 3 raw data files.
        - For L0B processing, it takes a small subset of the L0A Apache Parquet dataframe.
        The default is False.
    lazy : bool
        Whether to perform processing lazily with dask.
        If lazy=True, it employed dask.array and dask.dataframe.
        If lazy=False, it employed pandas.DataFrame and numpy.array.
        The default is True.
    single_netcdf : bool
        Whether to concatenate all raw files into a single L0B netCDF file.
        If single_netcdf=True, all raw files will be saved into a single L0B netCDF file.
        If single_netcdf=False, each raw file will be converted into the corresponding L0B netCDF file.
        The default is True.

    """
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

    columns_names_temporary = ["time", "epoch_time", "TO_BE_PARSED"]

    column_names = [
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
        # 'list_particles',
        "raw_drop_concentration",
        "raw_drop_average_velocity",
        "raw_drop_number",
    ]

    ##------------------------------------------------------------------------.
    #### - Define reader options

    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = ";"

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

        # Retrieve time in datetime format
        df["time"] = dd.to_datetime(df["time"], format="%Y%m%d-%H%M%S")
        df_time = df[["time"]]

        # Split the last column (into the 38 variable fields)
        df_to_parse = df["TO_BE_PARSED"].str.split(";", expand=True, n=99)

        # Add names to columns (exlcude last 3: raw_drop_concentration, raw_drop_average_velocity, raw_drop_number)
        df_to_parse_dict_names = dict(
            zip(column_names, list(df_to_parse.columns)[: len(column_names) - 3])
        )
        for i in range(len(column_names) - 3, len(list(df_to_parse.columns))):
            df_to_parse_dict_names[i] = i

        df_to_parse.columns = df_to_parse_dict_names

        # Remove char from rain intensity
        df_to_parse["rainfall_rate_32bit"] = df_to_parse[
            "rainfall_rate_32bit"
        ].str.lstrip("b'")

        # Remove spaces on weather_code_metar_4678 and weather_code_nws
        df_to_parse["weather_code_metar_4678"] = df_to_parse[
            "weather_code_metar_4678"
        ].str.strip()
        df_to_parse["weather_code_nws"] = df_to_parse["weather_code_nws"].str.strip()

        # Add the comma on the raw_drop_concentration, raw_drop_average_velocity and raw_drop_number
        if lazy:
            apply_kwargs = {"meta": (None, "object")}
        else:
            apply_kwargs = {}
        df_raw_drop_concentration = (
            df_to_parse.iloc[:, 35:67]
            .apply(lambda x: ",".join(x.dropna().astype(str)), axis=1, **apply_kwargs)
            .to_frame("raw_drop_concentration")
        )
        df_raw_drop_average_velocity = (
            df_to_parse.iloc[:, 67:99]
            .apply(lambda x: ",".join(x.dropna().astype(str)), axis=1, **apply_kwargs)
            .to_frame("raw_drop_average_velocity")
        )

        df_raw_drop_number = (
            df_to_parse.iloc[:, 99]
            .squeeze()
            .str.replace(r"(\w{3})", r"\1,", regex=True)
            .str.rstrip("'")
            .to_frame("raw_drop_number")
        )

        # Concat all dataframe togheter
        if lazy:
            concat_kwargs = {"ignore_unknown_divisions": True}
        else:
            concat_kwargs = {}
        df = dd.concat(
            [
                df_time,
                df_to_parse.iloc[:, :35],
                df_raw_drop_concentration,
                df_raw_drop_average_velocity,
                df_raw_drop_number,
            ],
            axis=1,
            **concat_kwargs
        )

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
        ]

        df = df.drop(todrop, axis=1)

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    files_glob_pattern = "*.csv*"

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
        # Custom arguments of the reader
        files_glob_pattern=files_glob_pattern,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
    )


if __name__ == "__main__":
    main()
