========
Overview
========

DISDRODB: A global database of raindrop size distribution observations

.. warning::
    This document is not complete !

    Currently under development.

    Do not use it now.

Motivation
================

The raindrop size distribution (DSD) describes the concentration and size distributions of raindrops in a volume of air. It is a crucial piece of  information to model the propagation of microwave signals through the atmosphere (key for telecommunication and weather radar remote sensing calibration), to improve microphysical schemes in numerical weather prediction models, and to understand land surface processes (rainfall interception, soil erosion).

The need for understanding the DSD spatio-temporal variability has led scientists all around the globe to “count the drops” by deploying DSD recording instruments known as disdrometers. Numerous measurement campaigns have been run by various meteorological services, national agencies (e.g. the NASA Precipitation Measurement Mission - PMM - Science Team), and university research groups. However, only a small fraction of those data is easily accessible. Data are stored in disparate formats with poor documentation, making them difficult to share, analyse, compare and re-use.  Additionally, very limited software is currently publicly available for DSD processing.


Structure of the project
========================


Here is the structure of the project :

| 📁 disdrodb/
| ├── 📁 L0/    *Raw data*
|     ├── 📁 configs
|     	├── 📁 <type_of_devices>/   *e.g OTT_Parsivel*
|     		├── 📜 \*.yml   *Define attributes formats*
|     		├── 📜 readme.md  *Describes config files*
|     ├── 📁 manuals
|       ├── 📜 \*.pdf  *Disdrometers documentation*
|     ├── 📁 readers|     	
|     	├── 📁 <type_of_reader>/
|           ├── 📜 \*.py *Official readers to transform raw data into standardized netCDF4 files.*
|     ├── 📁 templates
|     	├── 📁 <sensor_name>/
|     		├── 📜 \*.py *Templates for readers development.*
|       ├── 📜 reader_template.py *Template file to start developing a new reader*
|     ├── 📜 auxiliary.py *Mapping dictionary for ARM and DIVEN standards*
|     ├── 📜 check_configs.py : Sensor configs validator
|     ├── 📜 check_metadata.py : Metadata validator (unused ?)
|     ├── 📜 check_standards.py : Standard validator
|     ├── 📜 template_tools.py : Helper to create format station readers
|     ├── 📜 io.py : Core functions to read write files / folders
|     ├── 📜 L0A_processing.py : Core function to process raw data files to L0A format (Parquet)
|     ├── 📜 L0B_processing.py : Core function to process raw data files to L0B format (netCDF)
|     ├── 📜 L0_processing.py : Core function to process raw data files to L0A and L0B formats
|     ├── 📜 metadata.py : Create or read metadata for readers core functions
|     ├── 📜 standards.py : Implement functions to encode the L0 sensor specifications defined in L0.configs
|     ├── 📜 utils_nc.py : Utilty function to process raw netCDF data files of specific institute/networks
|     ├── 📜 issue.py : Issue file management to exclude erroneous timestamps or time periods while reading and processing the raw data
| ├── 📁 L1/
|     ├── to do
| ├── 📁 L2/
|     ├── to do
| ├── 📁 pipepline/
|   ├── 📜 utils_cmd.py : Trigger L0A and L0B processing for one specific reader
|   ├── 📜 \*.py : Scripts to run the full pipepline
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





