#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 03:10:48 2022

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
    # - In all files, the datalogger voltage hasn't the delimeter,
    #   so need to be split to obtain datalogger_voltage and rainfall_rate_32bit

    column_names = [
        "time",
        "id",
        "datalogger_temperature",
        "datalogger_voltage",
        "rainfall_rate_32bit",  # Intensity
        "rainfall_accumulated_32bit",
        "weather_code_synop_4680",
        "weather_code_synop_4677",
        "reflectivity_32bit",
        "mor_visibility",
        "laser_amplitude",
        "number_particles",
        "sensor_temperature",
        "sensor_heating_current",
        "sensor_battery_voltage",
        "sensor_status",
        "rainfall_amount_absolute_32bit",
        "Debug_data",
        "raw_drop_concentration",
        "raw_drop_average_velocity",
        "raw_drop_number",
        "All_0",
    ]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    # reader_kwargs['delimiter'] = ''

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
    reader_kwargs["na_values"] = ["na", "", "error", "NA"]

    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    # Cast all to string
    reader_kwargs["dtype"] = str

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

        # Drop Debug_data and All_0
        df = df.drop(
            columns=[
                "All_0",
                "Debug_data",
                "datalogger_voltage",
                "datalogger_temperature",
                "id",
            ]
        )

        # If raw_drop_number is nan, drop the row
        col_to_drop_if_na = [
            "raw_drop_concentration",
            "raw_drop_average_velocity",
            "raw_drop_number",
        ]
        df = df.dropna(subset=col_to_drop_if_na)

        # Drop rows with less than 4096 char on raw_drop_number
        df = df.loc[df["raw_drop_number"].astype(str).str.len() == 4096]

        # - Convert time column to datetime
        df["time"] = dd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")

        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    files_glob_pattern = "*.dat*"

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
