#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:02:42 2021

@author: kimbo
"""

# Metadata creation

# 1) Folder creation
# 2) JSON creation
# 3) JSON read
# 4) Loop JSON for every sensor

import json

import os
from sensor import Sensor
from campaign import Campaign

# File path
json_path = "/SharedVM/Campagne/ltnas3/Raw/EPFL_Roof_2008/EPFL_Roof_2008.json"
# json_path = "/SharedVM/Campagne/ltnas3/Raw/attribtues_template.json"

# Opening JSON file
f = open(json_path)
 
# returns JSON object as
# a dictionary
data = json.load(f)

data_campaign = []

for k,v in data['campaing'].items():
    data_campaign.append(v)

c = Campaign(data_campaign)


 
# Iterating through the json
# print(data)
    
# for i in data['device']:
#     print(i)
 
# Closing file
f.close()