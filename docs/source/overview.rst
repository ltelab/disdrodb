========
Overview
========

DISDRODB: A global database of raindrop size distribution observations


Motivation
================

The raindrop size distribution (DSD) describes the concentration and size 
distributions of raindrops in a volume of air. 
It is a crucial piece of  information to model the propagation of microwave signals
through the atmosphere (key for telecommunication and weather radar remote sensing calibration),
to improve microphysical schemes in numerical weather prediction models, and 
to understand land surface processes (rainfall interception, soil erosion).

The need for understanding the DSD spatio-temporal variability has led scientists
all around the globe to “count the drops” by deploying DSD recording instruments
known as disdrometers.
Numerous measurement campaigns have been run by various meteorological services, 
national agencies (e.g. the NASA Precipitation Measurement Mission - PMM - Science Team), 
and university research groups. 
However, only a small fraction of those data is easily accessible. 
Data are stored in disparate formats with poor documentation, making them 
difficult to share, analyse, compare and re-use. 

Additionally, very limited software is currently publicly available for DSD processing.





Software Structure
========================


Here below is described the current structure of the software:

| 📁 data/ : *Sample data*
| 📁 disdrodb/
| ├── 📁 l0 : Contains the software to produce the DISDRODB L0 products   
|     ├── 📁 configs : Contains the specifications of various types of disdrometers
|     	├── 📁 `<sensor_name>` : e.g. OTT_Parsivel, OTT_Parsivel2, Thies_LPM, RD_80
|     		├── 📜 \*.yml  : YAML files defining sensor characteristics (e.g. diameter and velocity bins)
|     ├── 📁 manuals 
|       ├── 📜 \*.pdf: Official disdrometers documentation
|     ├── 📁 readers
|     	├── 📁 `<data_source>` : e.g. GPM, ARM, EPFL, ...
|           ├── 📜 \<reader_name>.py : Readers to transform raw data into DISDRODB L0 products
|     ├── 📁 scripts : Contains a set of python scripts to be called from the terminal to launch the L0 processing 
|         ├── 📜 run_disdrodb_l0_station.py : Script launching the L0 processing for a specific station
|         ├── 📜 run_disdrodb_l0 : Script launching the L0 processing for specific portion of the DISDRODB archive
|         ├── 📜 *.py
|     ├── 📜 check_configs.py : Contain functions checking the sensor configs YAML files
|     ├── 📜 check_metadata.py : Contain functions checking the metadata YAML files
|     ├── 📜 check_standards.py : Contain functions checking that DISDRODB standards are met
|     ├── 📜 io.py : Core functions to read/write files and create/remove directories
|     ├── 📜 issue.py : Code to manage the issues YAML files and exclude erroneous timesteps during L0 processing
|     ├── 📜 l0a_processing.py : Contain the functions to process raw data files to L0A format (Parquet)
|     ├── 📜 l0b_processing.py : Contain the functions to process raw data files to L0B format (netCDF4)
|     |── 📜 l0b_concat.py : Contain the functions to concat multiple L0B files into a single L0B netCDF
|     ├── 📜 l0b_processing.py : Contain the functions to run the DISDRODB L0 processing
|     |── 📜 l0_reader.py : Contain the functions to check and retrieve the DISDRODB readers
|     ├── 📜 metadata.py : Code to read/write the metadata YAML files  
|     ├── 📜 standards.py : Contain the functions to encode the L0 sensor specifications defined in L0.configs
|     ├── 📜 summary.py : Contain the functions to define a summary for each station
|     ├── 📜 template_tools.py : Helpers to create DISDRODB readers
| ├── 📁 l1/
|     ├── Code not yet implemented. It will contain software to homogenize and quality check DISDRODB L0 products
| ├── 📁 l2/
|     ├── Code not yet implemented. It will contain software to produce DISDRODB L2 products (i.e. DSD parameters, ...)
| ├── 📁 api/
| ├── 📁 utils/
|   ├── 📜 logger.py : Logger functions
|   ├── 📜 scripts.py : Utility functions to run python scripts into the terminal
|   ├── 📜 netcdf.py : Utilty function to check and merge/concat multiple netCDF4 files
| 📁 docs/ *Documentation (generated with sphinx)*
| 📁 tutorials
   ├── 📜 reader_preparation.ipynb : Jupyter Notebook template to start developing a new reader
| 📜 .gitignore
| 📜 LICENSE
| 📜 CONTRIBUTING.md
| 📜 README.md
| 📜 requirements.txt





