#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:43:34 2022

@author: ghiggi
"""
import yaml 
from disdrodb.L0.issue import create_issue_yml
#--------------------------------------------------
# Open desired timestamps.yml
src_fpath = "/home/ghiggi/Projects/disdrodb/dev/src_timestamps.yml" 
with open(src_fpath, "r") as f:
    dict_issue_timestamps = yaml.safe_load(f)

#--------------------------------------------------
# Write desired timestamps.yml 
# # TODO: 
# - write same format as input yaml 
# - pyaml nested flow style  (for nested collections)
# default_flow_style args
# - False use block style output
# - True use inline output
# --> Solution: https://stackoverflow.com/questions/51976149/mixing-block-and-flow-formatting-in-yaml-with-python
dst_fpath = "/home/ghiggi/Projects/disdrodb/dev/dst_timestamps.yml" 
create_issue_yml(dst_fpath,
                 timestamp=dict_issue_timestamps.get("timestamp", None),
                 time_period=dict_issue_timestamps.get("time_period", None)
                )

#--------------------------------------------------
# TO DEVELOP THE CORRECT FLOW STYLE  
timestamp_dict = {"timestamp": ['2018-12-07 14:15','2018-12-07 14:17','2018-12-07 14:19', '2018-12-07 14:25']}

time_period_dict = {"time_period": [['2018-08-01 12:00:00', '2018-08-01 14:00:00'],
                                    ['2018-08-01 15:44:30', '2018-08-01 15:59:31'], 
                                    ['2018-08-02 12:44:30', '2018-08-02 12:59:31']]
                    }

with open(fpath, 'w') as f:
    yaml.safe_dump(timestamp_dict, f,
                   default_flow_style=None)
    yaml.safe_dump(time_period_dict, f, #
                   default_flow_style=None)

#--------------------------------------------------