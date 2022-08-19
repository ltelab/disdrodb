=========================
Readers
=========================


DISDRODB supports reading and loading data from many input file formats and schemes. The following sections describe the different way data can be loaded, requested, or added to the DISDRODB project.



Available Readers
==================

.. note::
    RL : Could be nice to have a function that returns all readers 


.. code-block:: bash

	from disdrodb import available_readers
	available_readers()
	


Add a new reader
==================


.. note::
    RL : Here we should prepare a fake project with sample data to give example for every steps. It should be a very light and simple dataset. 



**Step 1 : Create the file structure for the reader.**

Go under the reader folder and add yours (or use an exsiting one).

This folder must be named with the name of this institution. 

.. note::
    RL : we should formalize that ? How to name the folders to have consistency. Currently, there are institution names and country names. 




**Step 2 : Get the template python scripts.**

Copy paste the templates/l0_parser_dev_template into the folder proviously created. 

This file will be used as basis for your devlopment. 

.. note::
    RL : Here again, we should define a nomenclature  



**Step 3 : Prepare the raw and processes file tree.** 

The input raw folder tree must correspond to : 

| DISDRODB/
| ├── Raw/
|    ├── Name of the institution/
|       ├── Name ot the campaign/
|           ├── data
|               ├── ID of the station/ 
|                  ├── \*.\*  : raw file
|           ├── info        
|           ├── issue
|               ├── <ID of the station>.yml           
|           ├── metedata
|               ├── <ID of the station>.yml      


The output folder tree must correspond to : 

| DISDRODB/
| ├── Processed/
|    ├── Name of the institution/
|       ├── Name ot the campaign/
|           ├── homogenized data
|               ├── ID of the station/ 
|                  ├── \*.paquet
|                  ├── \*.nc 
|           ├── metedata
|               ├── <ID of the station>.yml   



.. note::
    RL :  This is a proposal. If we can avoid abbreviations such as l0, l1 and use standard nomenclature. 




**Step 4 : Export - Transform - Load your raw data into Apache Parquet**

The template you copied contains a first part where the developer can define how the row file will be read (eg. separator, header, ect). This part must be adapted to match your row file characteristics. 

The **df_sanitizer_fun** function must also be modified to add the data frame transformation and cleaning process.  This function is essential since it will shape the parquet output file. 

The script can be run via command line as follow :

.. code-block::

       python <python file path> <../DISDRODB/Raw/<Name of the institution>/<Name ot the campaign>>  <../DISDRODB/Processed/<Name of the institution>> -l0 true -l1 false -f true



to continue... 














