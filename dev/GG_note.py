#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 15:03:53 2021

@author: ghiggi
"""
##----------------------------------------------------------------------------.
## Infer Arrow schema from pandas
# schema = pa.Schema.from_pandas(df)

## dtype = {'col': pd.api.types.CategoricalDtype(['a', 'b', 'c'])}

##----------------------------------------------------------------------------.
### Dask Dataframe  
# schema = "infer"
# overwrite = force 
# partition_on 
# row_group_size = 100000

##----------------------------------------------------------------------------.
# read_csv 
# - Dask dataframe tries to infer the dtype of each column by reading a 
#   sample from the start of the file (or of the first file if it’s a glob). 
#   Usually this works fine, but if the dtype is different later in the file 
#   (or in other files) this can cause issues. 
#   For example, if all the rows in the sample had integer dtypes, 
#   but later on there was a NaN, then this would error at compute time.
#   To fix this, provide explicit dtypes for the offending columns using the dtype keyword.
 

