========
Overview
========

DISDRODB: A global database of raindrop size distribution observations


Motivation
================

The raindrop size distribution (DSD) describes the concentration and size distributions of raindrops in a volume of air. It is a crucial piece of  information to model the propagation of microwave signals through the atmosphere (key for telecommunication and weather radar remote sensing calibration), to improve microphysical schemes in numerical weather prediction models, and to understand land surface processes (rainfall interception, soil erosion).

The need for understanding the DSD spatio-temporal variability has led scientists all around the globe to “count the drops” by deploying DSD recording instruments known as disdrometers. Numerous measurement campaigns have been run by various meteorological services, national agencies (e.g. the NASA Precipitation Measurement Mission - PMM - Science Team), and university research groups. However, only a small fraction of those data is easily accessible. Data are stored in disparate formats with poor documentation, making them difficult to share, analyse, compare and re-use.  Additionally, very limited software is currently publicly available for DSD processing.


Structure of the project
========================


Here is the structure of the project :

| 📁 disdrodb/
| ├── 📁 L0 : Contains the software to produce the DISDRODB L0 products   
|     ├── 📁 configs : Contains the specifications of various types of disdrometers
|     	├── 📁 `<sensor_name>` : e.g. OTT_Parsivel, Thies_LPM, RD80
|     		├── 📜 \*.yml  : YAML files defining sensor characteristics (e.g. diameter and velocity bins)
|     ├── 📁 manuals 
|       ├── 📜 \*.pdf: Official disdrometers documentation
|     ├── 📁 readers
|     	├── 📁 `<data_source>` : e.g. GPM, ARM, EPFL, ...
|           ├── 📜 \reader_<campaign_name>.py : Readers to transform raw data into DISDRODB L0 products
|       ├── 📜 reader_preparation.ipynb : Jupyter Notebook template to start developing a new reader
|     ├── 📜 auxiliary.py : Mapping dictionary for some `data_source` standards (e.g. ARM)*
|     ├── 📜 check_configs.py : Sensor configs validator
|     ├── 📜 check_metadata.py : Metadata validator
|     ├── 📜 check_standards.py : Standards validator
|     ├── 📜 template_tools.py : Helpers to create station readers
|     ├── 📜 io.py : Core functions to read/write files and create/remove directories
|     ├── 📜 L0A_processing.py : Core function to process raw data files to L0A format (Parquet)
|     ├── 📜 L0B_processing.py : Core function to process raw data files to L0B format (netCDF4)
|     ├── 📜 L0_processing.py : Core function to process raw data files to L0A and L0B formats
|     ├── 📜 metadata.py : Create or read metadata files  
|     ├── 📜 standards.py : Implement functions to encode the L0 sensor specifications defined in L0.configs
|     ├── 📜 utils_nc.py : Utilty function to process raw netCDF4 data files of specific `data_source`
|     ├── 📜 issue.py : Issue file management to exclude erroneous timestamps or time periods while reading and processing the raw data
| ├── 📁 L1/
|     ├── Code not yet implemented. It will contain software to homogenize and quality check DISDRODB L0 products
| ├── 📁 L2/
|     ├── Code not yet implemented. It will contain software to produce DISDRODB L2 products (i.e. DSD parameters, ...)
| ├── 📁 pipeline/
|   ├── 📜 utils_cmd.py : Trigger L0A and L0B processing for specific L0 readers
|   ├── 📜 \*.py : Scripts to process data of specific `data_source`
| ├── 📁 api/
| ├── 📁 uils/
|   ├── 📜 logger.py : Logger functions
| 📁 docs/ *Documentation (generated with sphinx)*
| 📁 data/ *Sample data*
| 📜 .gitignore
| 📜 LICENSE
| 📜 CONTRIBUTING.md
| 📜 README.md
| 📜 requirements.txt





