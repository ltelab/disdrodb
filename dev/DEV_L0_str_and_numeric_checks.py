#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:42:14 2022

@author: ghiggi
"""                          
##----------------------------------------------------------------------------. 
#### Character EDA 
filepath = file_list[0]
str_reader_kwargs = reader_kwargs.copy() 
str_reader_kwargs['dtype'] = str # or object 
df = read_raw_data(filepath, column_names=None,  
                   reader_kwargs=str_reader_kwargs, lazy=False)

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


from disdrodb.check_standards import get_field_nchar_dict
from disdrodb.check_standards import get_field_ndigits_dict
from disdrodb.check_standards import get_field_ndigits_decimals_dict
from disdrodb.check_standards import get_field_ndigits_natural_dict

sensor_name = "Parsivel"
 
infer_df_column_names(df, sensor_name=sensor_name)

from disdrodb.dev_tools import print_df_columns_unique_values


idx = 20
print_df_columns_unique_values(df, column_indices=idx)
string = df.iloc[1,idx]
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

def get_possible_keys(dict_options, desired_value): 
    list_key_match = []
    for k, v in dict_options.items():
        if v == desired_value:
            list_key_match.append(k) 
    set_key_match = set(list_key_match)
    return set_key_match   

def search_possible_columns(string, sensor_name):
    dict_digits = get_field_ndigits_dict(sensor_name)
    dict_nchar_digits = get_field_nchar_dict(sensor_name)   
    dict_decimal_digits = get_field_ndigits_decimals_dict(sensor_name)  
    dict_natural_digits = get_field_ndigits_natural_dict(sensor_name)
    set_digits = get_possible_keys(dict_digits, get_ndigits(string))
    set_nchar = get_possible_keys(dict_nchar_digits, get_nchar(string))
    set_decimals = get_possible_keys(dict_decimal_digits, get_decimal_ndigits(string))
    set_natural = get_possible_keys(dict_natural_digits, get_natural_ndigits(string))
    possible_keys = set_digits.intersection(set_nchar, set_decimals, set_natural)
    possible_keys = list(possible_keys)
    return possible_keys

def infer_df_column_names(df, sensor_name, row_idx = 1):
    dict_possible_columns = {}
    for i, column in enumerate(df.columns):
        print(i)
        # Get string array 
        arr = df.iloc[:,i]
        arr = np.asarray(arr).astype(str)
        if not arr_has_constant_nchar(arr):
            print("Column",i, "has non-unique number of characters")
            continue 
        # Subset a single string 
        string = arr[row_idx]
        possible_columns = search_possible_columns(string, sensor_name=sensor_name)
        dict_possible_columns[i] = possible_columns
    return dict_possible_columns



####--------------------------------------------------------------------------.
#### Character checks 
import numpy as np 
def arr_has_constant_nchar(arr):
    arr = np.asarray(arr)
    if arr.dtype.char not in ['O', 'U']:
        raise TypeError("Expecting object (O) or string (U) dtype.")
    if arr.dtype.char == 'O': 
        arr = arr.astype(str)
    if arr.dtype.char != 'U':
       raise TypeError("Expecting string (U) dtype.")
    # Get number of characters (include .)              
    str_nchars = np.char.str_len(arr)   
    str_nchars_unique = np.unique(str_nchars)
    if len(str_nchars_unique) != 1: 
        return False #         raise ValueError("Non-unique string length !")     
    else:
        return True 
             
 
def str_is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def str_is_not_number(string):
    return not str_is_number(string) 

def str_is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
    
def str_has_decimal_digits(string):
    if len(string.split(".")) == 2: 
        return True 
    else:
        return False 
        
def get_decimal_ndigits(string): 
    if str_has_decimal_digits(string):
        return len(string.split(".")[1])
    else:
        return 0
    
def get_natural_ndigits(string): 
    if str_is_integer(string):
        return len(string)
    if str_has_decimal_digits(string): 
        return len(string.split(".")[0])
    else:
        return 0 
    
def get_ndigits(string): 
    if str_is_not_number(string):
        return 0 
    if str_has_decimal_digits(string): 
        return len(string) - 1 # remove . 
    else: 
        return len(string) 

def get_nchar(string):
    return len(string)





##----------------------------------------------------------------------------.
#### Numeric EDA
filepath = file_list[0]
df = read_raw_data(filepath, column_names=column_names,  
                   reader_kwargs=reader_kwargs, lazy=False)
 
from disdrodb.dev_tools import print_df_summary_stats
print_df_summary_stats(df)

# def checks_type = {  # between, isin 
#         

## TODO: 
# Check that min is < expected max 
# Check that max is > expected min 
# - Require removing na flag values (i.e. -99, ....)

###----------------------------------------------------------------------------.
# A function that makes numeric and character checks 
def check_reader_and_columns(filepath, 
                             column_names,  
                             reader_kwargs, 
                             lazy=False):
    # Define kwargs to have all columns as dtype 
    str_reader_kwargs = reader_kwargs.copy() 
    str_reader_kwargs['dtype'] = str # or object
    # Load numeric df 
    numeric_df = read_raw_data(filepath, column_names=column_names,  
                               reader_kwargs=reader_kwargs, lazy=lazy)
    # Load str df 
    str_df = read_raw_data(filepath, column_names=column_names,  
                        reader_kwargs=str_reader_kwargs, lazy=lazy)
    #-------------------------------------------------------------------------.
    # TODO all checks
    # - Option to raise Error when incosistency encountered 
    # - Option to add all errors in a list 
    # - Option to print (verbose) 
    # - List column names analyzed
    # - List column names not analyzed 
    # - List column names without inconsistencies 
    

# from disdrodb.io import col_dtype_check
# col_dtype_check(df, filename, verbose)

# def col_dtype_check(df, file_path, verbose = False):
#     dtype_max_digit = get_field_n_digits_dict()
#     dtype_range_values = get_field_value_range_dict()
#     ignore_colums_range = [
#         'FieldN',
#         'FieldV',
#         'RawData'
#         ]
#     for col in df.columns:
#         try:
#             # Check if all nan
#             if not df[col].isnull().all():
#                 # Check if data is longer than default in declared in dtype_max_digit
#                 if not df[col].astype(str).str.len().max() <= dtype_max_digit[col]:

                    # Check if data is in default range declared in dtype_range_values
                #if not df[col].between(dtype_range_values[col][0], dtype_range_values[col][1]).all():

                    
                # Check exact length for FieldN, FieldV and RawData
                # if col == 'FieldN' or col == 'FieldV':
                #     # if df[col].astype(str).str.len().min() =! dtype_max_digit[col][0] or df[col].astype(str).str.len().max() =! dtype_max_digit[col][0]:
                #     if df[col].astype(str).str.len().min() != dtype_max_digit[col][0] or df[col].astype(str).str.len().max() != dtype_max_digit[col][0]:
                #         msg = f'File: {file_path}, the values {col} are not in range: %s' % (dtype_max_digit[col][0].tolist())
                #         if verbose:
                #             print(msg)
                #         logger.warning(msg)