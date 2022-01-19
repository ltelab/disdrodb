#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 15:49:35 2022

@author: ghiggi
"""
### FIRST RULE: LOGGING STUFFS IS THE LAST STUFF TO CODE !!!!

#### I avoid use of this please ... or show me an example where is necessary 
df = df.dropna(thresh = (len(df.columns) - 10), how = 'all')  
df = df.replace({"na": np.nan, "nan": np.nan, "OK": 0, 'OK"': 0})    

#### Employ the L0_parser_dev_template.py for each campaign (in template folder)

#### Create a folder where placing all the parser dev codes (so that I can also test on ltenas3)

### Push every day !!!! 
# - Ping me so that I review code 

### Add conventions and useful stuffs to Guidelines.py 

####--------------------------------------------------------------------------.
#### TODO CODING 

## 2. Make /scripts/run_EPFL_processing.py work !

## 3. Launch full parser_TICINO_2018 on ltenas3 

## 4. Final review on parser_TICINO_2018 with me 

## 5. Add also other parser to /disdrodb/readers/EPFL following TICINO_2018 template 

## 6. Solve TODO in io.py/get_file_list

## 7. Correct create_directory_structure 
# .../processed/campaign_name>
# - L0
# - L1
# - metadata 
# - info 
# - <station_id> directories currently created must be removed ! 

## 7. Continue with other parsers 

####--------------------------------------------------------------------------.
#### TODO DATA 

# 3. Download GPM 