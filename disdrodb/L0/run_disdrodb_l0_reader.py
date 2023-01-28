##################################################
## Wrapper to run readers by command lines
##################################################
## Author: Régis Longchamp
##################################################
import click
import os 
import dask
from disdrodb.L0 import click_l0_readers_options
from dask.distributed import Client, LocalCluster

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator
@click.command()
@click.argument("data_source", metavar="<data_source>")
@click.argument("reader_name", metavar="<reader_name>")
@click_l0_readers_options  # get default arguments
def run_reader_cmd(**kwargs):
    """Wrapper to run reader functions by command lines.

    Parameters
    ----------
    data_source : str
        Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)
    reader_name : str
        Campaign name
    raw_dir : str
        The directory path where all the raw content of a specific campaign is stored.
        The path must have the following structure:
            <...>/DISDRODB/Raw/<data_source>/<campaign_name>'.
        Inside the raw_dir directory, it is required to adopt the following structure:
        - /data/<station_id>/<raw_files>
        - /metadata/<station_id>.yaml
        Important points:
        - For each <station_id> there must be a corresponding YAML file in the metadata subfolder.
        - The <campaign_name> must semantically match between:
           - the raw_dir and processed_dir directory paths;
           - with the key 'campaign_name' within the metadata YAML files.
        - The campaign_name are expected to be UPPER CASE.
    processed_dir : str
        The desired directory path for the processed DISDRODB L0A and L0B products.
        The path should have the following structure:
            <...>/DISDRODB/Processed/<data_source>/<campaign_name>'
        For testing purpose, this function exceptionally accept also a directory path simply ending
        with <campaign_name> (i.e. /tmp/<campaign_name>).
    l0a_processing : bool
      Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.
      The default is True.
    l0b_processing : bool
      Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.
      The default is True.
    keep_l0a : bool
        Whether to keep the L0A files after having generated the L0B netCDF products.
        The default is False.
    force : bool
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    parallel : bool 
        If True, the files are processed simultanously in multiple processes.
        Each process will use a single thread to avoid issues with the HDF/netCDF library.
        By default, the number of process is defined with os.cpu_count(). 
        However, you can customize it by typing: DASK_NUM_WORKERS=4 run_disdrodb_l0_reader 
        If False, the files are processed sequentially in a single process. 
        If False, multi-threading is automatically exploited to speed up I/0 tasks.
    debugging_mode : bool
        If True, it reduces the amount of data to process.
        - For L0A processing, it processes just 3 raw data files.
        - For L0B processing, it processes only the first 100 rows of 3 L0A files.
        The default is False.
    single_netcdf : bool
        Whether to concatenate all raw files into a single L0B netCDF file.
        If single_netcdf=True, all raw files will be saved into a single L0B netCDF file.
        If single_netcdf=False, each raw file will be converted into the corresponding L0B netCDF file.
        The default is True.
    """ 
    from disdrodb.L0.L0_processing import run_reader
    
    # If parallel=True, set dask environment 
    parallel = kwargs.get("parallel")
    if parallel: 
        # Set HDF5_USE_FILE_LOCKING to avoid going stuck with HDF 
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" 
        # Retrieve the number of process to run 
        available_workers = os.cpu_count()
        num_workers = dask.config.get('num_workers', available_workers)
        # Create dask.distributed local cluster
        cluster = LocalCluster(n_workers = num_workers,           
                               threads_per_worker=1,  
                               processes=True, 
                               # memory_limit='8GB',
                               # silence_logs=False,
                               )
        client = Client(cluster)
        
    # Run the processing 
    run_reader(**kwargs)
    

if __name__ == "__main__":
    
    run_reader_cmd()
    
    