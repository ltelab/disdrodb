=================================
Structure of the project
=================================


Proposed Project structure
================================


.. note::
    RL :This a basis of discussion. I took the terms "homogenization", "production" and "api" from the initial proposal



.. note::
    RL : I did not rename files, but I think it worth thinking to more relevant names.


Here is the structure of the project : 

| disdrodb/
| ├── homogenization/
|     ├── io.py
|     ├── L0_proc.py
|     ├── L1_proc.py
|     ├── issue.py  
|     ├── processing.py
|     ├── readers/
|         ├── <type_of_reader>/
|             ├── *.py 
|     ├── development readers/
|         ├── template/
|             ├── reader_template.py 
|         ├── <type_of_reader>/
|             ├── *.py 
|     ├── configs/
|         ├── <type_of_devices>/
|             ├── *.yml 
|     ├── utils/
|         ├── metadata.py  
|         ├── metadata_checks.py
|         ├── standards.py  
|         ├── check_standards.py 
|         ├── data_encodings.py
|         ├── configs_checks.py 
|         ├── dev_tool.py
|         ├── auxiliary.py  
|         ├── utils_nc.py
|         ├── utils.py
|     ├── data_sample/  
|     ├── tests/   
| ├── production/
|     ├── data_sample/ 
|     ├── tests/    
| ├── api/
|     ├── data_sample/  
|     ├── tests/ 
| ├── docs/
| .gitignore
| LICENSE
| CONTRIBUTING.md
| README.md
| requirements.txt

Note that this proposal describes only the reorganization of the file within the repository, not theirs names.  



Current Project structure 
================================
    
Here is an overview of the project structure : 

| disdrodb/
| ├── configs/
| │   ├── <type_of_devices>/
| │   │   ├── *.yml 
| ├── dev/
| │   ├── readers/
| │   │   ├── <type_of_reader>/
| │   │   │   ├── *.py 
| │   ├── configs_checks.py  
| │   ├── metadata_checks.py  
| ├── L0/
| │   ├── auxiliary.py  
| │   ├── issue.py  
| │   ├── processing.py  
| │   ├── utils_nc.py  
| ├── L1/
| │   ├── utils.py  
| ├── readers/
| │   ├── <type_of_reader>/
| │   │   ├── *.py 
| ├── utils  
| │   ├── parser.py  
| ├── check_standards.py
| ├── data_encodings.py
| ├── dev_tool.py
| ├── io.py
| ├── L0_proc.py
| ├── L1_proc.py
| ├── logger.py
| ├── metadata.py
| ├── standard.py
| docs/
| tests/
| data/
| templates/
| scripts/
| .gitignore
| LICENSE
| CONTRIBUTING.md
| README.md
| requirements.txt



Files description : 

.. note::
    RL : work in progress


**readers/\*.py** : Current readers (parsers) to transform raw data into a standardize Apache parquet file.  *RL : should use "pasrer" or "reader" -> to rename*

**L0A/auxiliary.py** : Define dictionary mapping for ARM and DIVEN standard *RL : to move into specific reader or utils ? *

**L0A/issue.py** : Create an Yml issue file to exclue time related error while reading raw data *RL : to move into  utils ? *

**L0A/processing.py** :  *RL : is this file used ? *

**L0A/utils_nc.py** :  Define specific functions for ARM and DIVEN standard *RL : to move into specific reader or utils ? *

**templates/\*.py** : Template to create new pasrser

**scripts/\*.py** :  Script to batch processing compains 

**check_standards.py** : Data quality function *RL : to move into  utils ?  rename ?*

**data_encodings.py** : Define the encoding of parquet column *RL : to move into  utils ?  rename ? Not used in any readers ?*

**dev_tool.py** : Functions to help the developer to create a format specific reader *RL : to move into  utils ?  rename ? Not used in any readers ?*

**io.py** : Functions to translate raw data into into a standardize Apache parquet file *RL : to move into utils, rename ? *

**L0_proc.py** : Process the translation from raw data into into a standardize Apache parquet file *Move into L0A*

**L1_proc.py** : Process the translation from standardize Apache parquet file into netCDF. *Move into L0B*

**logger.py** : Create log file. *Move into utils*

**metadata.py** : Create, reader metadata fo reader *Move into utils ?*

**standard.py** : Retrive devices characteritics *Move into utils ?*

