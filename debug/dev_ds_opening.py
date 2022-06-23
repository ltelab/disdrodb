#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:41:26 2022

@author: ghiggi
"""
import time 

  #### Open netCDFs
  file_list = sorted(file_list)
  try:
      ds = xr.open_mfdataset(file_list)
  except ValueError:   
      # Temporaray solution to skip netCDF consistency 
      print('Error, no monotonic global indexes along dimension time, than ignore check')
      # ds = xr.open_mfdataset(file_list, combine='nested', compat='override')
      ds = xr.open_mfdataset(file_list, 
                             combine='nested', 
                             compat='override', 
                             # chunks = {'time':1440}
                             )
  except Exception as e:
      msg = f"Error in read netCDF dataset. The error is: \n {e}"
      raise RuntimeError(msg)
      
file_list = sorted(file_list)
t_i = time.time()
ds = xr.open_mfdataset(file_list[0:100], 
                        combine='nested', 
                        compat='override', 
                        # chunks = {'time':1440}
                        )
t_f = time.time()
print(t_f - t_i)


t_i = time.time()
ds = xr.open_mfdataset(file_list[0:1000], 
                       concat_dim="time", 
                       combine='nested', 
                       compat='no_conflicts', # or override ... to not account of changing attributes
                       coords = "minimal",
                       chunks = {'time':5000},
                       )
t_f = time.time()
print(t_f - t_i) # 108 s

t_i = time.time()
ds = xr.open_mfdataset(file_list[0:1000], 
                       concat_dim="time", 
                       combine='nested', 
                       compat='no_conflicts', # or override ... to not account of changing attributes
                       coords = "minimal",
                       chunks = {'time':5000},
                       parallel = True, 
                       )
t_f = time.time()
print(t_f - t_i) # 159 s 


#---------------------------------
t_i = time.time()
ds = xr.open_mfdataset(file_list[0:1000])
t_f = time.time()
print(t_f - t_i) # 32 s

#---------------------------------
t_i = time.time()
ds = xr.open_mfdataset(file_list[0:1000], 
                       chunks = {'time':5000}, 
                       )
t_f = time.time()
print(t_f - t_i) # 33 s

#---------------------------------
t_i = time.time()
ds = xr.open_mfdataset(file_list[0:1000], 
                       chunks = {'time':5000}, 
                       parallel = True, 
                       )
t_f = time.time()
print(t_f - t_i) # SLOW !!!
#---------------------------------

t_i = time.time()
ds = xr.open_mfdataset(file_list[0:10000], 
                       chunks = {'time':5000}, 
                       )
t_f = time.time()
print(t_f - t_i)  

#---------------------------------


# ValueError: Resulting object does not have monotonic global indexes along dimension time 
# - Multithreading enabled 
import dask 
from dask.distributed import Client
client = Client(processes=True) # n_workers=2, threads_per_worker=2
client.ncores()
client.nthreads()

t_i = time.time() 

@dask.delayed
def open_dataset_delayed(fpath):
    ds = xr.open_dataset(fpath, chunks={"time": -1}, engine="netcdf4")
    return ds


list_ds_delayed = []
for fpath in file_list[0:1000]:
    # print(fpath)
    # ds = xr.open_dataset(fpath, chunks={"time": -1}, engine="netcdf4")
    list_ds_delayed.append(open_dataset_delayed(fpath))

list_ds = dask.compute(list_ds_delayed)[0]
t_f = time.time()
print(t_f - t_i) 

# The concatenate step is never supposed to handle partially overlapping coordinates
# Check non overlapping coordinates 

ds1 = xr.concat(list_ds, dim="time", data_vars="all")

for i, ds in enumerate(list_ds): 
    print(i)
    list_ds[i] = ds.drop('base_time')

t_f = time.time()
print(t_f - t_i) 

# combine_by_coords use xr.concat
# combine_nested use xr.merge 

# combine_nested manage to work with overlapping coordinates 

# The concatenate step is never supposed to handle partially overlapping coordinates
# xr.concat 
# - With the default parameters, xarray will load some coordinate variables into memory to compare them between datasets.
# - Default parameters may be prohibitively expensive if you are manipulating your dataset lazily
# -- data_vars:  if data variable with no dimension changing over files ... need to specify "all"

  
# compat='no_conflicts' is only available when combining xarray objects with merge --> combine_nested

## Time diff 
def ensure_monotonic_dimension(fpaths, list_ds, dim="time"):
    """Ensure that a list of xr.Dataset has a monotonic increasing dimension."""
    # Get dimension values array (and associated list_ds/xr.Dataset indices)    
    dim_values = np.concatenate([ds[dim].values for ds in list_ds])
    list_index = np.concatenate([np.ones(len(ds[dim]))*i for i, ds in enumerate(list_ds)])
    ds_index = np.concatenate([np.arange(0, len(ds[dim])) for i, ds in enumerate(list_ds)])
       
    # Identify index where start decreasing or duplicated
    diff_dim_values = np.diff(dim_values)
    indices_decreasing = np.where(diff_dim_values.astype(float) <= 0)[0] + 1
  
    if len(indices_decreasing) > 0: 
        idx_start_decreasing = indices_decreasing[0] # or duplicate
        
        # Find last timestep that is lower or equal to that timestep
        idx_restart_increase = np.max(np.where(dim_values <= dim_values[idx_start_decreasing])[0])
    
        # Indices to remove 
        idx_to_remove = np.arange(idx_start_decreasing, idx_restart_increase+1)
        list_index_bad = list_index[idx_to_remove]
        ds_index_bad = ds_index[idx_to_remove]
        dim_values_bad = dim_values[idx_to_remove]
        
        # Get dictionary of values/idx which are duplicated (or lead to no monotonic dimension) 
        dict_ds_bad_values = {k: dim_values_bad[np.where(list_index_bad==k)[0]] for k in np.unique(list_index_bad)}
        dict_ds_bad_idx = {k: ds_index_bad[np.where(list_index_bad==k)[0]] for k in np.unique(list_index_bad)}
        
        # Print message  
        for ds_idx_bad, bad_values in dict_ds_bad_values.items(): 
            fpath = fpaths[ds_idx_bad]
            msg = f"In {fpath}, dropping {dim} values {bad_values} to ensure monotonic {dim} dimension."
            print(msg)
        
        # Remove non-unique and not  da 
        for ds_idx_bad, bad_idx in dict_ds_bad_idx.items():
            ds = list_ds[ds_idx_bad]
            list_ds[bad_ds_idx] = ds.drop_isel({dim: bad_idx})
        
        # Iterative check 
        list_ds = ensure_monotonic_dimension(fpaths, list_ds, dim=dim)
        
    # Return list of xr.Dataset with monotonic dimension 
    return list_ds
 
   
    

ds_index = [0,1,2,3,0,1,2,3,4]
list_index = [0,0,0,0,1, 1, 1,1, 1]
dim_values = [0,1,5,5,5, 5, 6,7,8]
list_index = np.array(list_index)
dim_values = np.array(dim_values)
ds_index = np.array(ds_index)

# Check duplicated timesteps 


# Get index with non monotonic timesteps 








