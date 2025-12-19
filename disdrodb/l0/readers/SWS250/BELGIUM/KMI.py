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
"""DISDRODB reader for KMI Biral SWS250 sensors."""

import pandas as pd

from disdrodb.l0.l0_reader import is_documented_by, reader_generic_docstring
from disdrodb.l0.l0a_processing import read_raw_text_file


def parse_spectrum_line_to_string(line):
    """Parse one M... spectrum line into a zero-padded string with 21 values."""
    # Define number of velocity bins values expected
    n_cols = 21

    # Split spectrum line
    parts = line.split(",")

    # Check line validity
    n_values = len(parts)
    if n_values > n_cols:
        raise ValueError(f"Unexpected number of velocity bins: {n_values}.")

    # Strip  'M' from first bin
    parts[0] = parts[0].replace(":00M", "")

    # Strip last two letter from last value
    parts[-1] = parts[-1][:3]

    # Define list of values
    values = [int(x) for x in parts]
    if len(values) < n_cols:
        values.extend([0] * (n_cols - len(values)))
    values = values[:n_cols]

    # Define comma-separated string
    string = ",".join(str(v) for v in values)
    return string


def parse_spectrum_block(lines):
    """Parse an M-block into a fixed (16 x 21) matrix."""
    n_values = len(lines)
    if n_values != 16:
        raise ValueError(f"Unexpected number of diameter bins: {n_values}.")
    raw_drop_number_string = ",".join([parse_spectrum_line_to_string(line) for line in lines])
    return raw_drop_number_string


def build_spectrum_block(group):
    """Create SWS250 raw spectrum string."""
    try:
        return pd.Series(
            {
                "raw_drop_number": parse_spectrum_block(group["spectrum_line"].tolist()),
            },
        )
    except Exception:
        return pd.Series({"raw_drop_number": "NaN"})


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

    # - Skip rows with badly encoded data
    reader_kwargs["encoding_errors"] = "replace"

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
    # Identify rows with data
    df_params = df[df["TO_PARSE"].str.count(",") == 23]

    # Identify rows with spectrum matrix
    df_spectrum = df[df["TO_PARSE"].str.startswith(":00M")]
    if len(df_spectrum) == 0:
        raise ValueError("No spectrum available.")

    df_spectrum = df_spectrum["TO_PARSE"].str.rsplit(",", expand=True, n=2)
    df_spectrum.columns = ["spectrum_line", "date", "time"]
    df_spectrum["datetime"] = pd.to_datetime(
        df_spectrum["date"] + " " + df_spectrum["time"],
        format="%d/%m/%Y %H:%M:%S",
    )

    # Define groups
    # - Mark new group when time gap > 10 s
    is_new_group = (df_spectrum["datetime"].diff().dt.total_seconds() > 10).fillna(True)
    group_id = is_new_group.cumsum()
    # - Assign the first datetime of each group
    df_spectrum["group_time"] = df_spectrum.groupby(group_id)["datetime"].transform("first")

    # Group spectrum by timesteps
    df_raw_drop_number = (
        df_spectrum.groupby("group_time", as_index=False)
        .apply(build_spectrum_block, include_groups=False)
        .reset_index(drop=True)
    )

    # Retrieve 1-min data
    # - Split by ; delimiter (before raw drop number)
    df_data = df_params["TO_PARSE"].str.split(",", expand=True)

    # - Assign column names
    names = [
        "date",
        "time",
        "sws250",
        "sensor_id",
        "sample_interval",
        "mor_visibility_5min",  # remove unit and ensure in meters  !
        "weather_code_synop_4680",
        "past_weather1",
        "past_weather2",
        "obstruction_status",
        "weather_code_metar_4678",
        "precipitation_rate",
        "mor_visibility",  # remove unit and ensure in meters  !
        "total_extinction_coefficient",  # [km-1]
        "transmissometer_extinction_coefficient",  # [km-1]
        "back_scatter_extinction_coefficient",  # [km-1]
        "sensor_temperature",  # [degrees]  or air_temperature ?
        "ambient_light_sensor_signal",  # [cd/m2] # ALS
        "sensor_status",
        "number_particles",
        "precipitation_accumulated",  # [mm] over sample_interval
        "ambient_light_sensor_signal_status",
        "date1",
        "time1",
    ]
    df_data.columns = names

    # Clean out variables
    df_data["mor_visibility_5min"] = df_data["mor_visibility_5min"].str.replace("M", "")
    df_data["mor_visibility"] = df_data["mor_visibility"].str.replace("M", "")
    df_data["sensor_temperature"] = df_data["sensor_temperature"].str.replace("C", "")
    df_data["ambient_light_sensor_signal"] = df_data["ambient_light_sensor_signal"].str.replace("+99999", "NaN")

    # Define datetime
    df_data["datetime"] = pd.to_datetime(df_data["date1"] + " " + df_data["time1"], format="%d/%m/%Y %H:%M:%S")

    # Merge df_data on df_raw_drop_number
    # TODO list
    # - should we aggregate variables to 5 min temporal resolution
    #   to match raw_drop_number
    # - should we infill df_raw_drop_number with 0 when no time every 5 min?
    df = pd.merge_asof(
        df_raw_drop_number,
        df_data,
        left_on="group_time",
        right_on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("10s"),  # max difference allowed
    )

    # Define final time
    # TODO list
    # - which time should we take as final time?
    # - raw_drop_number time is end of measurement interval right?
    df["time"] = df["group_time"]

    # Drop columns not agreeing with DISDRODB L0 standards
    columns_to_drop = [
        "group_time",
        "date",
        "sws250",
        "sample_interval",
        "sensor_id",
        "date1",
        "time1",
        "datetime",
    ]
    df = df.drop(columns=columns_to_drop)
    return df
