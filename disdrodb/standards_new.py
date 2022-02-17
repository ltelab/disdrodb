#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:06:35 2022

@author: ghiggi
"""
import os
import yaml
import logging
logger = logging.getLogger(__name__)

def get_configs_dir(sensor_name): 
    dir_path = os.path.dirname(__file__)
    config_dir_path = os.path.join(dir_path, "configs") 
    config_sensor_dir_path = os.path.join(config_dir_path, sensor_name) 
    if not os.path.exists(config_sensor_dir_path): 
        print("Available sensor_name are {}:".format(sorted(os.listdir(config_dir_path))))
        raise ValueError("The config directory for sensor {} is not available. ".format(sensor_name))
    return config_sensor_dir_path

def get_available_sensor_name(): 
    dir_path = os.path.dirname(__file__)
    config_dir_path = os.path.join(dir_path, "configs") 
    # TODO: here add checks that contains all required yaml file 
    return sorted(os.listdir(config_dir_path))
                  
def get_diameter_bins_dict(sensor_name): 
    config_sensor_dir_path = get_configs_dir(sensor_name)
    fpath = os.path.join(config_sensor_dir_path, "diameter_bins.yml")
    if not os.path.exists(fpath): 
        msg = "'diameter_bins.yml' not available in {}".format(config_sensor_dir_path)
        logger.exception(msg)
        raise ValueError(msg)
    # TODO: 
    # Check dict contains center, bounds and width keys 
    
    # Open diameter bins dictionary 
    with open(fpath, 'r') as f:
        d = yaml.safe_load(f)
    return d 

def get_velocity_bins_dict(sensor_name): 
    config_sensor_dir_path = get_configs_dir(sensor_name)
    fpath = os.path.join(config_sensor_dir_path, "velocity_bins.yml")
    if not os.path.exists(fpath):
        msg = "'velocity_bins.yml' not available in {}".format(config_sensor_dir_path)
        logger.exception(msg)
        raise ValueError(msg)
    # Open diameter bins dictionary 
    with open(fpath, 'r') as f:
        d = yaml.safe_load(f)
    return d 


def get_diameter_bin_center(sensor_name): 
    diameter_dict = get_diameter_bins_dict(sensor_name) 
    return diameter_dict['center']


def get_diameter_bin_lower(sensor_name): 
    diameter_dict = get_diameter_bins_dict(sensor_name) 
    lower_bounds = [v[0] for v in diameter_dict['bounds'].values()]
    return lower_bounds 


def get_diameter_bin_upper(sensor_name): 
    diameter_dict = get_diameter_bins_dict(sensor_name) 
    upper_bounds = [v[1] for v in diameter_dict['bounds'].values()]
    return upper_bounds 


def get_diameter_bin_width(sensor_name): 
    diameter_dict = get_diameter_bins_dict(sensor_name) 
    return diameter_dict['width']


def get_velocity_bin_center(sensor_name): 
    velocity_dict = get_velocity_bins_dict(sensor_name) 
    return velocity_dict['center']


def get_velocity_bin_lower(sensor_name): 
    velocity_dict = get_velocity_bins_dict(sensor_name) 
    lower_bounds = [v[0] for v in velocity_dict['bounds'].values()]
    return lower_bounds 


def get_velocity_bin_upper(sensor_name): 
    velocity_dict = get_velocity_bins_dict(sensor_name) 
    upper_bounds = [v[1] for v in velocity_dict['bounds'].values()]
    return upper_bounds 


def get_velocity_bin_width(sensor_name): 
    velocity_dict = get_velocity_bins_dict(sensor_name) 
    return velocity_dict['width']

    
def get_raw_field_nbins(sensor_name): 
    diameter_dict = get_diameter_bins_dict(sensor_name) 
    velocity_dict = get_velocity_bins_dict(sensor_name) 
    n_d = len(diameter_dict['center'])
    n_v = len(velocity_dict['center'])
    nbins_dict = {"FieldN": n_d,
                  "FieldV": n_v,
                  "RawData": n_d*n_v,
                 }
    return nbins_dict
