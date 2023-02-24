#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import xarray as xr


def _create_summary(
    ds: xr.Dataset,
    processed_dir: str,
    station_name: str,
) -> None:
    """Create L0 summary statistics and save it into the station info YAML file.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset.
    processed_dir : str
        Output file path
    station_name : str
        Station ID
    """

    ###-----------------------------------------------------------------------.
    # Get the sensor name
    ds.attrs.get("sensor_name")

    # Initialize dictionary
    stats_dict = {}

    # Infer the sampling interval looking at the difference between timesteps
    dt, counts = np.unique(np.diff(ds.time.values), return_counts=True)
    dt_most_frequent = dt[np.argmax(counts)]
    dt_most_frequent = dt_most_frequent.astype("m8[s]")
    inferred_sampling_interval = dt_most_frequent.astype(int)
    stats_dict["inferred_sampling_interval"] = inferred_sampling_interval

    # Number of years, months, days, minutes
    time = ds.time.values
    n_timesteps = len(time)
    n_minutes = inferred_sampling_interval / 60 * n_timesteps
    n_hours = n_minutes / 60
    n_days = n_hours / 24

    stats_dict["n_timesteps"] = n_timesteps
    stats_dict["n_minutes"] = n_minutes
    stats_dict["n_hours"] = n_hours
    stats_dict["n_days"] = n_days

    # Add start_time and end_time
    start_time = pd.DatetimeIndex(time[[0]])
    end_time = pd.DatetimeIndex(time[[-1]])
    years = np.unique([start_time.year, end_time.year])
    if len(years) == 1:
        years_coverage = str(years[0])
    else:
        years_coverage = str(years[0]) + "-" + str(years[-1])

    stats_dict["years_coverage"] = years_coverage
    stats_dict["start_time"] = start_time[0].isoformat()
    stats_dict["end_time"] = end_time[0].isoformat()

    ###-----------------------------------------------------------------------.
    # TODO: Create and save image with temporal coverage
    # --> Colored using quality flag from sensor_status if available ?

    ###-----------------------------------------------------------------------.
    # TODO STATISTICS
    # --> Requiring deriving stats from raw spectrum

    # diameter_min, diameter_max, diameter_sum

    # Total rain events

    # Total rainy minutes

    # Total dry minutes

    # Number of dry/rainy minutes

    ###-----------------------------------------------------------------------.
    # Save to info.yaml
    info_path = os.path.join(processed_dir, "info", station_name + ".yml")
    with open(info_path, "w") as f:
        yaml.dump(stats_dict, f, sort_keys=False)

    return None


# TODO: Add command line script
# _create_summary(
#     ds=ds,
#     processed_dir=processed_dir,
#     station_name=station_name,
# )
