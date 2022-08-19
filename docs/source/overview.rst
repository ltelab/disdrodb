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

| disdrodb/
| ├── L0/
|     ├── configs
|     	├── <type_of_devices>/
|     		├── *.yml
|     ├── readers
|     	├── <type_of_reader>/
|     		├── *.py 
|     ├── templates
|     	├── <type_of_reader>/
|     		├── *.py
|     ├── auxiliary.py 
|     ├── check_configs.py 
|     ├── check_metadata.py 
|     ├── check_standards.py 
|     ├── dev_tools.py 
|     ├── io.py 
|     ├── L0A_processing.py
|     ├── L0B_processing.py 
|     ├── L0_processing.py 
|     ├── metadata.py 
|     ├── standards.py 
|     ├── utils_cmd.py 
|     ├── utils_nc.py 
|     ├── issue.py 
| ├── L1/
|     ├── to do
| ├── L2/
|     ├── to do
| ├── pipepline/
| ├── api/
| ├── uils/
| docs/
| data/
| refences/
| .gitignore
| LICENSE
| CONTRIBUTING.md
| README.md
| requirements.txt


L0 Files description
=====================

**configs/\*.yml** : todo

**readers/\*.py** : Current readers (parsers) to transform raw data into a standardize Apache parquet file.  *RL : should use "pasrer" or "reader" -> to rename*

**templates/\*.py** : Template to create new pasrser

**auxiliary.py** : Define dictionary mapping for ARM and DIVEN standard

**check_config.py** : todo

**check_metadata.py** : todo

**check_standards.py** : Data quality function *RL : to move into  utils ?  rename ?*

**dev_tool.py** : Functions to help the developer to create a format specific reader *RL : to move into  utils ?  rename ? Not used in any readers ?*

**io.py** : Functions to translate raw data into into a standardize Apache parquet file *RL : to move into utils, rename ? *

**L0A_processing.py** : Process the translation from raw data into into a standardize Apache parquet file *Move into L0A*

**L0B_processing.py** : Process the translation from standardize Apache parquet file into netCDF. *Move into L0B*

**L0_processing.py** :  *RL : is this file used ? *

**metadata.py** : Create, reader metadata fo reader *Move into utils ?*

**standards.py** : Retrive devices characteritics *Move into utils ?*

**utils_cmd** : todo

**utils_nc** : todo * Define specific functions for ARM and DIVEN standard *RL : to move into specific reader or utils ? *

**issue.py** : Create an Yml issue file to exclue time related error while reading raw data *RL : to move into  utils ? *



