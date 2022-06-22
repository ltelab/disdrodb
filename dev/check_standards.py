#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:53:49 2022

@author: ghiggi
"""
import os
import yaml

from disdrodb.standards_new import get_diameter_bins_dict
from disdrodb.standards_new import get_velocity_bins_dict
from disdrodb.standards_new import get_available_sensor_name
from disdrodb.standards import get_variables_dict
from disdrodb.standards import get_sensor_variables
from disdrodb.standards import get_units_dict
from disdrodb.standards import get_explanations_dict

sensor_name = "OTT_Parsivel"
sensor_name = "OTT_Parsivel2"
sensor_name = "Thies_LPM"

get_available_sensor_name()

get_diameter_bins_dict(sensor_name)
get_velocity_bins_dict(sensor_name)

get_variables_dict(sensor_name)
get_sensor_variables(sensor_name)

get_units_dict(sensor_name)
get_explanations_dict(sensor_name)


# with open("/home/ghiggi/velocity_bins.yml", "w") as f:
#     yaml.dump(dd, f, sort_keys=False)

#
