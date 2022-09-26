#!/usr/bin/env python3

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


def reader(
    raw_dir,
    processed_dir,
    l0a_processing=True,
    l0b_processing=True,
    keep_l0a=False,
    force=False,
    verbose=False,
    debugging_mode=False,
    lazy=False,
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
    l0a_processing : bool
      Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
      The default is True.
    l0b_processing : bool
      Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
      The default is True.
    keep_l0a : bool
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
    column_names = [
        "date",
        "time",
        "status",
        "interval",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "n10",
        "n11",
        "n12",
        "n13",
        "n14",
        "n15",
        "n16",
        "n17",
        "n18",
        "n19",
        "n20",
        "RI",
        "RA",
        "RAT",
    ]

    ##------------------------------------------------------------------------.
    #### - Define reader options
    reader_kwargs = {}
    # - Define delimiter
    reader_kwargs["delimiter"] = "\\t"

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
    reader_kwargs["na_values"] = ["na", "", "error"]

    # - Define max size of dask dataframe chunks (if lazy=True)
    #   - If None: use a single block for each file
    #   - Otherwise: "<max_file_size>MB" by which to cut up larger files
    reader_kwargs["blocksize"] = None  # "50MB"

    # Skip header
    reader_kwargs["header"] = None

    # Skip first row as columns names
    reader_kwargs["skiprows"] = 1

    # - Define encoding
    reader_kwargs["encoding"] = "ISO-8859-1"

    ##------------------------------------------------------------------------.
    #### - Define facultative dataframe sanitizer function for L0 processing
    # - Enable to deal with bad raw data files
    # - Enable to standardize raw data files to L0 standards  (i.e. time to datetime)
    def df_sanitizer_fun(df, lazy=False):
        # Import dask or pandas
        if lazy:
            import dask.dataframe as dd
        else:
            import pandas as dd

        # Replace 'status' NaN with 0
        df["status"] = df["status"].fillna(0)

        # Replace all ',' with '.' in RI, RA, RAT
        df["RI"] = df["RI"].replace({",": "."}, regex=True)
        df["RA"] = df["RA"].replace({",": "."}, regex=True)
        df["RAT"] = df["RAT"].replace({",": "."}, regex=True)

        # Define time column
        df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
        df["time"] = dd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
        df = df.drop(columns=["date"])

        # Create raw_drop_concentration string
        bin_columns = [
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
            "n7",
            "n8",
            "n9",
            "n10",
            "n11",
            "n12",
            "n13",
            "n14",
            "n15",
            "n16",
            "n17",
            "n18",
            "n19",
            "n20",
        ]

        df["raw_drop_number"] = ""
        for c in bin_columns:
            df["raw_drop_number"] += df[c].astype(str) + ";"

        df = df.drop(columns=bin_columns)
        return df

    ##------------------------------------------------------------------------.
    #### - Define glob pattern to search data files in raw_dir/data/<station_id>
    files_glob_pattern = "*.txt"

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
