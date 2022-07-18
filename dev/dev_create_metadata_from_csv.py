#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 07:29:20 2022

@author: kimbo
"""

def create_metadata(fpath, data):
    """Create default YAML metadata file."""
    import yaml
    
    with open(fpath, "w+") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    

def create_list_path(df, folder_output_path):
    '''Return list of yaml file path from DSD metadata.csv dataframe'''
    # Dataframe for files
    df_struc = df.iloc[:,:3]
    df_struc.columns = ['inst','camp','id']
    df_struc = df_struc.fillna('missing_id')

    # df_struc['inst'].unique()

    # Create structure folder
    list_path = []

    import os

    for i, r in df_struc.iterrows():
        
        path = os.path.join(folder_output_path, r['inst'], r['camp'], 'metadata', r['id'] + '.yml')
        
        list_path.append(path)
        
        # if not isinstance(path, str):
        #     raise TypeError("'path' must be a strig.")
        # try:
        #     os.makedirs(path)
        # except FileExistsError:
        #     pass
    
    print("Found {} paths for metadata files".format(len(list_path)))
    
    return list_path

def create_list_meta(df):
    '''Return list for yaml dump'''
    # Remove useless columns
    df = df.iloc[:,3:]
    df = df.drop(columns = ['comment'])

    # Reset index
    df = df.reset_index(drop=True)
    
    list_meta = df.to_dict('records')
    
    print("Found {} metadata files".format(len(list_meta)))
    
    return list_meta


def create_metadata_files(list_path, list_meta):
    '''Create all the metadata files'''
    import os
    
    for i in range(len(list_meta)):
        os.makedirs(os.path.dirname(list_path[i]), exist_ok=True)
        create_metadata(list_path[i], list_meta[i])
    
    print(" - Metadata files creation complete! - ")
        
def clean_df(df):
    '''Clean the dataframe by comments into DSD metadata.csv'''
    # Drop roww without metadata (comments into the drive sheet)
    df = df.dropna(thresh=10)

    # Replace nan
    values = {"Id dispositivo": 'missing_id',"latitude": -9999, "longitude": -9999, "altitude": -9999}
    df = df.fillna(value=values)
    df = df.fillna('')
    
    # Assign dtype to columns
    col_dtype = {'latitude': 'float',
                'longitude': 'float',
                'altitude': 'float',}
    
    for col, dtype in col_dtype.items():
        # Replace , with .
        try:
            df[col] = df[col].str.replace(',','.')
        except AttributeError:
            # Not need to replace
            pass
        
        try:
            # df[col] = df[col].astype(dtype)
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            print('Errors on dtype conversion on column {} for {} dtype'.format(col, dtype))
        
        
    
    return df

def check_csv(source_path):
    '''Check if file in source_path is DSD metadata.csv'''
    try:
        df = pd.read_csv(source_path)
    except pd.errors.EmptyDataError:
        print("Something wrong with .csv, any metadata found, please check .csv file")
        raise SystemExit
    
    # Check if df has right columns number
    if len(df.columns) != 51:
        print("Something wrong with .csv, the file should have 51 columns, {} were found, please check .csv file".format(len(df.columns)))
        raise SystemExit
    
    print("File is Ok")
    
    return df

import pandas as pd
import click

# -------------------------------------------------------------------------.
# CLIck Command Line Interface decorator
# @click.command()  # options_metavar='<options>'
# @click.argument('source_path', type=click.File('rb'))
# @click.argument('folder_output_path')
def main(source_path,
         folder_output_path
         ):
    
    # # Path for DSD metadata.csv
    # source_path = '/home/kimbo/Downloads/DSD_metadata.csv'

    # # Output folder for the metadata
    # folder_output_path = '/home/kimbo/data/metadata_test'
    
    df = check_csv(source_path)

    df = clean_df(df)

    list_path = create_list_path(df, folder_output_path)

    list_meta = create_list_meta(df)

    # Check if list_path and list_meta have the same lenght
    if len(list_path) != len(list_meta):
        print('Something wrong with .csv')
        raise SystemExit

    create_metadata_files(list_path, list_meta)

if __name__ == '__main__':
    # main()
    main(source_path = '/home/kimbo/Downloads/DSD_metadata.csv',
             folder_output_path = '/home/kimbo/data/metadata_test'
             )




