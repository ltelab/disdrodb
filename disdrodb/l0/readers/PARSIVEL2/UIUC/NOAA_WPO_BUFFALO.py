# -----------------------------------------------------------------------------.
# DISDRODB reader for UIUC SCAMP OTT Parsivel2 (NOAA_WPO_BUFFALO campaign).
#
# Reads the consolidated daily CSV files produced from the raw per-timestep
# .MIS telegrams by disdrodb_prep/mis_to_daily_csv.py. Each CSV is
# ';'-delimited with one row per timestep and a header row. The three spectrum
# columns are ','-separated strings (32, 32 and 1024 values respectively).
#
# Place this file at:
#   disdrodb/l0/readers/PARSIVEL2/UIUC/NOAA_WPO_BUFFALO.py
# and set in the station metadata:  reader: "UIUC/NOAA_WPO_BUFFALO"
# -----------------------------------------------------------------------------.
"""DISDRODB reader for the UIUC SCAMP Parsivel2 (NOAA WPO Buffalo)."""

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
    #### Define column names (must match mis_to_daily_csv.py output order)
    column_names = [
        "time",
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
        "sensor_time",
        "sensor_date",
        "station_name",
        "station_number",
        "raw_drop_concentration",
        "raw_drop_average_velocity",
        "raw_drop_number",
    ]

    ##------------------------------------------------------------------------.
    #### Define reader options
    reader_kwargs = {}
    reader_kwargs["engine"] = "python"
    reader_kwargs["compression"] = "infer"  # handles .csv and .csv.gz
    reader_kwargs["delimiter"] = ";"
    reader_kwargs["header"] = None
    reader_kwargs["skiprows"] = 1  # skip the header row we wrote
    reader_kwargs["index_col"] = False
    reader_kwargs["on_bad_lines"] = "skip"
    reader_kwargs["na_values"] = ["na", "", "error", "NA", "-.-", "-9.999"]

    ##------------------------------------------------------------------------.
    #### Read the data
    df = read_raw_text_file(
        filepath=filepath,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        logger=logger,
    )

    ##------------------------------------------------------------------------.
    #### Adapt to DISDRODB L0 standards
    # 'time' is already written as UTC "%Y-%m-%d %H:%M:%S" from the filename.
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df[df["time"].notna()]

    # Drop redundant / constant identifier columns (kept in station metadata)
    columns_to_drop = [
        "sensor_time",
        "sensor_date",
        "station_name",
        "station_number",
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    return df
