#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:42:14 2022

@author: ghiggi
"""                          
##----------------------------------------------------------------------------.
#### Numeric EDA
filepath = file_list[0]
df = read_raw_data(filepath, column_names=column_names,  
                   reader_kwargs=reader_kwargs, lazy=False)
 
from disdrodb.dev_tools import print_df_summary_stats
print_df_summary_stats(df)

## TODO: 
# Check that min is < expected max 
# Check that max is > expected min 
# - Require removing na flag values (i.e. -99, ....)

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
get_n_digits(string)

####--------------------------------------------------------------------------.
#### Character checks 
def check_constant_nchar(arr):
    arr = np.asarray(arr)
    if arr.dtype != object: 
        arr = arr.astype(str)
    if arr.dtype.char != 'U':
        raise TypeError("Expecting str dtype.")
    # Get number of characters (include .)              
    str_nchars = np.char.str_len(arr)   
    str_nchars_unique = np.unique(str_nchars)
    if len(str_nchars_unique) != 1: 
        raise ValueError("Non-unique string length !")                  
 
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
        
def get_decimal_digits(string): 
    if str_has_decimal_digits(string):
        return len(string.split(".")[1])
    else:
        return 0
    
def get_integer_digits(string): 
    if str_is_integer(string):
        return len(string)
    if str_has_decimal_digits(string): 
        return len(string.split(".")[0])
    else:
        return 0 
    
def get_n_digits(string): 
    if str_is_not_number(string):
        return 0 
    if str_has_decimal_digits(string): 
        return len(string) - 1 # remove . 
    else: 
        return len(string) 
        

#### Numeric checks 


from disdrodb.io import col_dtype_check
col_dtype_check(df, filename, verbose)

def col_dtype_check(df, file_path, verbose = False):

    dtype_max_digit = get_field_n_digits_dict()
    dtype_range_values = get_field_value_range_dict()
    
    ignore_colums_range = [
        'FieldN',
        'FieldV',
        'RawData'
        ]
    
    try:
        df = df.compute()
    except AttributeError:
        #Not dask, so is pandas, ignore
        pass
    except Exception as e:
        msg = f'Error on read dataframe, error: {e}'
        if verbose:
            print(msg)
        logger.warning(msg)

    for col in df.columns:
        try:
            # Check if all nan
            if not df[col].isnull().all():
                # Check if data is longer than default in declared in dtype_max_digit
                if not df[col].astype(str).str.len().max() <= dtype_max_digit[col]:
                    msg1 = f'{col} has more than %s digits' % dtype_max_digit[col]
                    msg2 = f"File: {file_path}, the values {col} have too much digits (%s) in index: %s" % (dtype_max_digit[col][0], df.index[df[col].astype(str).str.len() >= dtype_max_digit[col][0]].tolist())
                    if verbose:
                        print(msg1)
                        print(msg2)
                    logger.warning(msg1)
                    logger.warning(msg2)
                    
                # Check if data is in default range declared in dtype_range_values
                if not df[col].between(dtype_range_values[col][0], dtype_range_values[col][1]).all():
                    msg = f'File: {file_path}, the values {col} in index are not in dtype range: %s' % (df.index[df[col].between(dtype_range_values[col][0], dtype_range_values[col][1]) == False].tolist())
                    if verbose:
                        print(msg)
                    logger.warning(msg)
                # Check exact length for FieldN, FieldV and RawData
                # if col == 'FieldN' or col == 'FieldV':
                #     # if df[col].astype(str).str.len().min() =! dtype_max_digit[col][0] or df[col].astype(str).str.len().max() =! dtype_max_digit[col][0]:
                #     if df[col].astype(str).str.len().min() != dtype_max_digit[col][0] or df[col].astype(str).str.len().max() != dtype_max_digit[col][0]:
                #         msg = f'File: {file_path}, the values {col} are not in range: %s' % (dtype_max_digit[col][0].tolist())
                #         if verbose:
                #             print(msg)
                #         logger.warning(msg)
            else:
                msg = f'File: {file_path}, {col} has all nan, check ignored'
                if verbose:
                    print(msg)
                logger.warning(msg)
            pass
                 
        except KeyError:
            if col in ignore_colums_range:
                msg = f'File: {file_path}, no range values for {col}, check ignored'
                if verbose:
                    print(msg)
                logger.warning(msg)
            pass
        except TypeError:
            msg = f'File: {file_path}, {col} is object, check ignored'
            if verbose:
                print(msg)
            logger.warning(msg)
            pass

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