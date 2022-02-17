#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:53:49 2022

@author: ghiggi
"""
import yaml
import os
from disdrodb.standards import get_OTT_Parsivel_diameter_bin_center 
from disdrodb.standards import get_OTT_Parsivel_diameter_bin_bounds
from disdrodb.standards import get_OTT_Parsivel_diameter_bin_width
from disdrodb.standards import get_OTT_Parsivel_velocity_bin_center 
from disdrodb.standards import get_OTT_Parsivel_velocity_bin_bounds
from disdrodb.standards import get_OTT_Parsivel_velocity_bin_width
from disdrodb.standards import get_ThiesLPM_diameter_bin_center
from disdrodb.standards import get_ThiesLPM_diameter_bin_bounds
from disdrodb.standards import get_ThiesLPM_diameter_bin_width 
from disdrodb.standards import get_ThiesLPM_velocity_bin_center
from disdrodb.standards import get_ThiesLPM_velocity_bin_bounds
from disdrodb.standards import get_ThiesLPM_velocity_bin_width

center = get_OTT_Parsivel_diameter_bin_center()
bounds = get_OTT_Parsivel_diameter_bin_bounds()
width = get_OTT_Parsivel_diameter_bin_width()


center = get_OTT_Parsivel_velocity_bin_center()
bounds = get_OTT_Parsivel_velocity_bin_bounds()
width = get_OTT_Parsivel_velocity_bin_width()

center = get_ThiesLPM_diameter_bin_center()
bounds = get_ThiesLPM_diameter_bin_bounds()
width = get_ThiesLPM_diameter_bin_width()

center = get_ThiesLPM_velocity_bin_center()
bounds = get_ThiesLPM_velocity_bin_bounds()
width = get_ThiesLPM_velocity_bin_width()



dict_center = {i: float(el) for i, el in enumerate(center)}
dict_bounds = {i: bounds[i,:].tolist() for i in range(len(bounds))}
dict_width = {i: float(el) for i, el in enumerate(width)}

dd = {}
dd['center'] = dict_center
dd['bounds'] = dict_bounds
dd['width'] = dict_width
with open("/home/ghiggi/velocity_bins.yml", 'w') as f:
    yaml.dump(dd, f, sort_keys=False)


from disdrodb.standards_new import get_diameter_bins_dict
from disdrodb.standards_new import get_velocity_bins_dict
sensor_name = "OTT_Parsivel"
diameter_dict = get_diameter_bins_dict(sensor_name)
velocity_dict = get_velocity_bins_dict(sensor_name)

sensor_name = "Thies_LPM"
diameter_dict = get_diameter_bins_dict(sensor_name)
velocity_dict = get_velocity_bins_dict(sensor_name)

from disdrodb.standards_new import get_available_sensor_name
get_available_sensor_name()
