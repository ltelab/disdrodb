#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:42:14 2022

@author: ghiggi
"""                 
from disdrodb.dev_tools import print_df_columns_unique_values         
sensor_name = "Parsivel"
##----------------------------------------------------------------------------. 
#### Character EDA 
filepath = file_list[0]
str_reader_kwargs = reader_kwargs.copy() 
str_reader_kwargs['dtype'] = str # or object 
df = read_raw_drop_number(filepath, column_names=None,  
                   reader_kwargs=str_reader_kwargs, lazy=False)

df = df_str
arr = df.iloc[:,1]
check_constant_nchar(arr) 

# Check only on a random (i.e. second) row 
string = arr[1]

nchar = len(string)      
str_is_not_number(string)
str_is_number(string)
str_is_integer(string)
str_has_decimal_digits(string) # check if has a comma and right digits 
get_decimal_digits(string)     # digits right of the comma 
get_integer_digits(string)     # digits left of the comma 
get_digits(string)
get_nchar(string)

dict_digits = get_field_ndigits_dict(sensor_name)
dict_nchar_digits = get_field_nchar_dict(sensor_name)   
dict_decimal_digits = get_field_ndigits_decimals_dict(sensor_name)  
dict_natural_digits = get_field_ndigits_natural_dict(sensor_name)

#----------------------------------------------------------------------------.
idx = 4
print_df_columns_unique_values(df, column_indices=idx)
string = str(df.iloc[1,idx])
string
column_names[idx]


get_ndigits(string)
get_nchar(string)
get_decimal_ndigits(string)
get_natural_ndigits(string)

get_possible_keys(dict_digits, get_ndigits(string))
get_possible_keys(dict_nchar_digits, get_nchar(string))

get_possible_keys(dict_decimal_digits, get_decimal_ndigits(string))
get_possible_keys(dict_natural_digits, get_natural_ndigits(string))
 
infer_df_column_names(df, sensor_name=sensor_name)

##----------------------------------------------------------------------------.
#### Numeric EDA
filepath = file_list[0]
df = read_raw_drop_number(filepath, column_names=column_names,  
                   reader_kwargs=reader_kwargs, lazy=False)
 
from disdrodb.dev_tools import print_df_summary_stats
print_df_summary_stats(df)

###----------------------------------------------------------------------------.
# A function that makes numeric and character checks 
# def check_reader_and_columns(filepath, 
#                              column_names,  
#                              reader_kwargs, 
#                              lazy=False):
#     # Define kwargs to have all columns as dtype 
#     str_reader_kwargs = reader_kwargs.copy() 
#     str_reader_kwargs['dtype'] = str # or object
#     # Load numeric df 
#     numeric_df = read_raw_drop_number(filepath, column_names=column_names,  
#                                reader_kwargs=reader_kwargs, lazy=lazy)
#     # Load str df 
#     str_df = read_raw_drop_number(filepath, column_names=column_names,  
#                         reader_kwargs=str_reader_kwargs, lazy=lazy)
#     #-------------------------------------------------------------------------.
#     # TODO all checks
#     # - List column names analyzed
#     # - List column names not analyzed 
#     # - List column names without inconsistencies 
    

