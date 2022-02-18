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

# Retrieve sensor standards and configs

# -----------------------------------------------------------------------------.
import os
import yaml
import logging

logger = logging.getLogger(__name__)


def get_configs_dir(sensor_name):
    """Retrieve configs directory."""
    dir_path = os.path.dirname(__file__)
    config_dir_path = os.path.join(dir_path, "configs")
    config_sensor_dir_path = os.path.join(config_dir_path, sensor_name)
    if not os.path.exists(config_sensor_dir_path):
        print(
            "Available sensor_name are {}:".format(sorted(os.listdir(config_dir_path)))
        )
        raise ValueError(
            "The config directory for sensor {} is not available. ".format(sensor_name)
        )
    return config_sensor_dir_path


def get_available_sensor_name():
    """Get available sensor_name."""
    dir_path = os.path.dirname(__file__)
    config_dir_path = os.path.join(dir_path, "configs")
    # TODO: here add checks that contains all required yaml file
    return sorted(os.listdir(config_dir_path))


def get_variables_dict(sensor_name):
    """Get a dictionary containing the variable name of the sensor field numbers."""
    config_sensor_dir_path = get_configs_dir(sensor_name)
    fpath = os.path.join(config_sensor_dir_path, "variables.yml")
    if not os.path.exists(fpath):
        msg = "'variables.yml' not available in {}".format(config_sensor_dir_path)
        logger.exception(msg)
        raise ValueError(msg)
    # Open diameter bins dictionary
    with open(fpath, "r") as f:
        d = yaml.safe_load(f)
    return d


def get_sensor_variables(sensor_name):
    """Get sensor variable names list."""
    return list(get_variables_dict(sensor_name).values())


def get_units_dict(sensor_name):
    """Get a dictionary containing the unit of each sensor variable."""
    config_sensor_dir_path = get_configs_dir(sensor_name)
    fpath = os.path.join(config_sensor_dir_path, "variable_units.yml")
    if not os.path.exists(fpath):
        msg = "'variable_units.yml' not available in {}".format(config_sensor_dir_path)
        logger.exception(msg)
        raise ValueError(msg)
    # Open diameter bins dictionary
    with open(fpath, "r") as f:
        d = yaml.safe_load(f)
    return d


def get_explanations_dict(sensor_name):
    """Get a dictionary containing the explanation of each sensor variable."""
    config_sensor_dir_path = get_configs_dir(sensor_name)
    fpath = os.path.join(config_sensor_dir_path, "variable_explanations.yml")
    if not os.path.exists(fpath):
        msg = "'variable_explanations.yml' not available in {}".format(
            config_sensor_dir_path
        )
        logger.exception(msg)
        raise ValueError(msg)
    # Open diameter bins dictionary
    with open(fpath, "r") as f:
        d = yaml.safe_load(f)
    return d


def get_diameter_bins_dict(sensor_name):
    """Get dictionary with sensor_name diameter bins information."""
    config_sensor_dir_path = get_configs_dir(sensor_name)
    fpath = os.path.join(config_sensor_dir_path, "diameter_bins.yml")
    if not os.path.exists(fpath):
        msg = "'diameter_bins.yml' not available in {}".format(config_sensor_dir_path)
        logger.exception(msg)
        raise ValueError(msg)
    # TODO:
    # Check dict contains center, bounds and width keys

    # Open diameter bins dictionary
    with open(fpath, "r") as f:
        d = yaml.safe_load(f)
    return d


def get_velocity_bins_dict(sensor_name):
    """Get velocity with sensor_name diameter bins information."""
    config_sensor_dir_path = get_configs_dir(sensor_name)
    fpath = os.path.join(config_sensor_dir_path, "velocity_bins.yml")
    if not os.path.exists(fpath):
        msg = "'velocity_bins.yml' not available in {}".format(config_sensor_dir_path)
        logger.exception(msg)
        raise ValueError(msg)
    # Open diameter bins dictionary
    with open(fpath, "r") as f:
        d = yaml.safe_load(f)
    return d


def get_diameter_bin_center(sensor_name):
    """Get diameter bin center."""
    diameter_dict = get_diameter_bins_dict(sensor_name)
    diameter_bin_center = list(diameter_dict["center"].values())
    return diameter_bin_center


def get_diameter_bin_lower(sensor_name):
    """Get diameter bin lower bound."""
    diameter_dict = get_diameter_bins_dict(sensor_name)
    lower_bounds = [v[0] for v in diameter_dict["bounds"].values()]
    return lower_bounds


def get_diameter_bin_upper(sensor_name):
    """Get diameter bin upper bound."""
    diameter_dict = get_diameter_bins_dict(sensor_name)
    upper_bounds = [v[1] for v in diameter_dict["bounds"].values()]
    return upper_bounds


def get_diameter_bin_width(sensor_name):
    """Get diameter bin width."""
    diameter_dict = get_diameter_bins_dict(sensor_name)
    diameter_bin_width = list(diameter_dict["width"].values())
    return diameter_bin_width


def get_velocity_bin_center(sensor_name):
    """Get velocity bin center."""
    velocity_dict = get_velocity_bins_dict(sensor_name)
    velocity_bin_center = list(velocity_dict["center"].values())
    return velocity_bin_center


def get_velocity_bin_lower(sensor_name):
    """Get velocity bin lower bound."""
    velocity_dict = get_velocity_bins_dict(sensor_name)
    lower_bounds = [v[0] for v in velocity_dict["bounds"].values()]
    return lower_bounds


def get_velocity_bin_upper(sensor_name):
    """Get velocity bin upper bound."""
    velocity_dict = get_velocity_bins_dict(sensor_name)
    upper_bounds = [v[1] for v in velocity_dict["bounds"].values()]
    return upper_bounds


def get_velocity_bin_width(sensor_name):
    """Get velocity bin width."""
    velocity_dict = get_velocity_bins_dict(sensor_name)
    velocity_bin_width = list(velocity_dict["width"].values())
    return velocity_bin_width


def get_raw_field_nbins(sensor_name):
    diameter_dict = get_diameter_bins_dict(sensor_name)
    velocity_dict = get_velocity_bins_dict(sensor_name)
    n_d = len(diameter_dict["center"])
    n_v = len(velocity_dict["center"])
    nbins_dict = {
        "FieldN": n_d,
        "FieldV": n_v,
        "RawData": n_d * n_v,
    }
    return nbins_dict


# -----------------------------------------------------------------------------.
