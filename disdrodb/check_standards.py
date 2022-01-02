#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------.
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
#-----------------------------------------------------------------------------.
import logging
logger = logging.getLogger(__name__)


def available_sensor_name():
    sensor_list = ['Parsivel', 'Parsivel2', 'ThiesLPM']
    return sensor_list

def check_sensor_name(sensor_name): 
    if not isinstance(sensor_name, str):
        logger.exception("'sensor_name' must be a string'")
        raise TypeError("'sensor_name' must be a string'")
    if sensor_name not in available_sensor_name():
        msg = f"Valid sensor_name are {available_sensor_name()}"
        logger.exception(msg)
        raise ValueError(msg)
    return 
 
def check_L0_column_names(x):
    # Allow TO_BE_SPLITTED, TO_BE_PARSED
    pass

def check_L0_standards(x):
    # TODO:
    pass

def check_L1_standards(x):
    # TODO:
    pass

def check_L2_standards(x): 
    # TODO:
    pass

def get_flags(device):
    if device == 'Parsivel':
        flag_dict = {
            'sensor_status': [  # TODO[KIMBO] : WHY 0 MISSING 
                1,
                2,
                3
            ], 
            'datalogger_error' : [ # TODO[KIMBO] : WHY 0 MISSING 
                1
            ],
            'error_code' : [   # TODO[KIMBO] : WHY 0 MISSING 
                1,
                2
            ]
            }
    return flag_dict

