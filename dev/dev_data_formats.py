#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:33:48 2022

@author: ghiggi
"""
import os
import yaml
from disdrodb.standards import get_sensor_variables
from disdrodb.check_standards import get_field_ndigits_natural_dict
from disdrodb.check_standards import get_field_ndigits_decimals_dict
from disdrodb.check_standards import get_field_ndigits_dict
from disdrodb.check_standards import get_field_nchar_dict
from disdrodb.check_standards import get_field_value_range_dict
from disdrodb.check_standards import get_field_flag_dict


sensor_name = "OTT_Parsivel"

n_natural = get_field_ndigits_natural_dict(sensor_name)
n_decimals = get_field_ndigits_decimals_dict(sensor_name)
n_digits = get_field_ndigits_dict(sensor_name)
n_chars = get_field_nchar_dict(sensor_name)
data_range = get_field_value_range_dict(sensor_name)
nan_flags = get_field_flag_dict(sensor_name)
variables = get_sensor_variables(sensor_name)

data_dict = {}
for var in variables:
    data_dict[var] = {}
    data_dict[var]["n_digits"] = n_digits.get(var, None)
    data_dict[var]["n_characters"] = n_chars.get(var, None)
    data_dict[var]["n_decimals"] = n_decimals.get(var, None)
    data_dict[var]["n_naturals"] = n_natural.get(var, None)
    data_dict[var]["data_range"] = data_range.get(var, None)
    data_dict[var]["nan_flags"] = nan_flags.get(var, None)


fpath = os.path.join(
    "/home/ghiggi/Projects/disdrodb/disdrodb/configs", sensor_name, "L0_data_format.yml"
)
with open(fpath, "w") as f:
    yaml.dump(data_dict, f, sort_keys=False)

from disdrodb.standards import get_data_format_dict

data_dict = get_data_format_dict(sensor_name)
data_dict
