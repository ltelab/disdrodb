#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:39:40 2021

@author: ghiggi
"""
import os
import argparse # prefer click ... https://click.palletsprojects.com/en/8.0.x/
import click
import time
import dask 
import dask.dataframe as dd
import numpy as np 
import xarray as xr 

from parsiveldb import XXXX
from parsiveldb import check_L0_standards
from parsiveldb import check_L1_standards
from parsiveldb import check_L2_standards
from parsiveldb import get_attrs_standards
from parsiveldb import get_dtype_standards
from parsiveldb import get_L1_encodings_standards
from parsiveldb import get_L1_chunks_standards

# A file for each instrument 
# TODO: force=True ... remove existing file 
#       force=False ... raise error if file already exists 
# Time execution 
print('- Ln, validation and test sets: {:.2f}s'.format(time.time() - t_i))
   

def main(base_dir, L0_processing=True, L1_processing=True, force=False, verbose=True):
    #-------------------------------------------------------------------------.
    ###############################
    #### Perform L0 processing ####
    ###############################
    if L1_processing: 
        if verbose:
            print("L0 processing of XXXX started")
        t_i = time.time()
        # Custom code to read from raw data     
        # - Define attributes 
        attrs = get_attrs_standards()
        attrs['Title'] = 
        attrs['Title'] = ...
        
        # - Define headers 
        columns = ['..., ...']
        check_valid_varname(columns)     
        
        # - Loop over files 
        
        
        # - Replace custom NA with standard flags 
        
        # - Write Parquet (with optmized dtype)
        dtypes_dict = get_dtype_standards()
        dtypes_dict = {{column: dtypes_dict[column] for column in columns}
                       
        # - Check Parquet standards 
        check_L0_standards(df)
        
        if verbose:
            print("L0 processing of XXXX ended in {}:.2f}".format(time.time() - t_i)))
    #-------------------------------------------------------------------------.
    ###############################
    #### Perform L1 processing ####
    ###############################
    if L1_processing: 
        if verbose:
            print("L1 processing of XXXX started")
        t_i = time.time()    
        # Check the L0 df is available 
        if not L0_processing:
            if not os.path.exists(df_fpath):
                raise ValueError("Need to run L0 processing. The {} file is not available.".format(df_fpath))
        df_path = '' # TBD
        df = dd.read_parquet(df_path)
        
        # - Conversion to xarray 
        ds = # TODO
        ds.attrs = attrs
        
        # - Save netcdf 
        L1_nc_fpath = '' # TBD
        encoding = get_L1_encodings_standards()
        chunks = get_L1_chunks_standards()
        ds.to_netcdf(L1_nc_fpath, encoding=encoding, chunks=chunks)
        
        if verbose:
            print("L1 processing of XXXX ended in {}:.2f}".format(time.time() - t_i)))
    #-------------------------------------------------------------------------.

# TODO: maybe found a better way --> click
# https://click.palletsprojects.com/en/8.0.x/ 

if __name__ == '__main__':
    main() # if using click     
    # Otherwise:     
    parser = argparse.ArgumentParser(description='L0 and L1 data processing')
    parser.add_argument('--base_dir', type=str)
    parser.add_argument('--L0_processing', type=str, default='True')
    parser.add_argument('--L1_processing', type=str, default='True')
    parser.add_argument('--force', type=str, default='False')                    
    
    L0_processing=True, L1_processing=True, force=False
    
    args = parser.parse_args()
    if args.force == 'True':
        force = True
    else: 
        force = False
    if args.L0_processing == 'True':
        L0_processing = True
    else: 
        L0_processing = False 
     if args.L1_processing == 'True':
        L1_processing = True
    else: 
        L1_processing = False   
        
    main(base_dir = base_dir, 
         L0_processing=L0_processing, 
         L1_processing=L1_processing,
         force=force)
 
