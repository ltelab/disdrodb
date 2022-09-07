=========================
Readers development
=========================


.. warning::
    This document is not complete !

    Currently under development.

    Do not use it now.

	Note that we use the words "parser" and "reader" interchangeably.

DISDRODB supports reading and loading data from many input file formats and schemes. The following sections describe the different way data can be loaded, requested, or added to the DISDRODB project.



Available Readers
########################

The following function returns the list of all readers.


.. code-block:: bash

	from disdrodb import available_readers
	available_readers()

.. warning::
    Not implemented yet !




Running a reader
########################

To execute a reader, run the following command :

.. code-block::

       python python_file_path raw_dir  processed_dir [parameters]



There are a couple of optional parameters that can added to the previous command :

* ``raw_dir`` : str - Directory path of raw file for a specific campaign.

	* The path should end with <campaign_name>.
	* Example raw_dir: '<...>/disdrodb/data/raw/<campaign_name>'.


* ``processed_dir`` : str - Desired directory path for the processed L0A and L0B products.

	* The path should end with <campaign_name> and match the end of raw_dir.
	* Example: '<...>/disdrodb/data/processed/<campaign_name>'.

* ``--l0a_processing`` : bool [ **true** |false] - Whether to launch processing to generate L0A Apache Parquet file(s) from raw data.



* ``--l0b_processing`` : bool [ **true** |false] - Whether to launch processing to generate L0B netCDF4 file(s) from L0A data.



* ``--keep_l0a`` : bool [true| **false** ] - Whether to keep the L0A files after having generated the L0B netCDF products.



* ``--force`` : bool [true| **false** ] - Whether to overwrite existing data.

	*  If True, overwrite existing data into destination directories.
	*  If False, raise an error if there are already data into destination directories.


* ``--verbose`` : bool [true| **false** ] -  Whether to print detailed processing information into terminal.



* ``--debugging_mode`` : bool [true| **false** ] -  If True, it reduces the amount of data to process.

	* For L0A processing, it processes just 3 raw data files.
	* For L0B processing, it takes a small subset of the L0A Apache Parquet dataframe.



* ``--lazy`` : bool [ **true** |false] - Whether to perform processing lazily with dask.

	* If lazy=True, it employed dask.array and dask.dataframe.
	* If lazy=False, it employed pandas.DataFrame and numpy.array.



* ``--single_netcdf`` : bool [ **true** |false] - Whether to concatenate all raw files into a single L0B netCDF file.

	* If single_netcdf=True, all raw files will be saved into a single L0B netCDF file.
	* If single_netcdf=False, each raw file will be converted into the corresponding L0B netCDF file.






Adding a new reader
########################

* `Step 1 <#step-1-set-the-folder-structure-for-raw-and-processed-datasets>`_ : Set the folder structure for raw and processed datasets
* `Step 2 <#step-2-read-and-analyse-the-data>`_ :  Read and analyse the data
* `Step 3 <#step-3-create-the-reader>`_ :  Create the reader


See also the step-by-step `tutorial <#adding-a-new-reader-tutorial>`_   that will demonstrate in detail all these steps with a sample lightweight dataset.


Step 1 : Set the folder structure for raw and processed datasets
*******************************************************************

The raw and processed data folder must follow strictly the following structure:

Raw data folder
======================

| 📁 DISDRODB/
| ├── 📁 Raw/
|    ├── 📁 NAME_OF_INSTITUTION_OR_COUNTRY/
|       ├── 📁 NAME_OF_CAMPAIGN/
|           ├── 📁 data
|               ├── 📁 <ID of the station>/
|                  ├── 📜 \*.\*  : raw file
|           ├── 📁 info
|           ├── 📁 issue
|               ├── 📜 <ID of the station>.yml
|           ├── 📁 metadata
|               ├── 📜 <ID of the station>.yml


.. note::
	Guidelines for the **Name of the institution or country** folder :

	* We use the institution name when campaign data spans more than 1 country.
	* We use country when all campaigns (or sensor networks) are inside a given country.

.. note::
    For each folder in the /data directory (for each station) there must be an equally named **\*.yml** file in the metadata folder. This file contains information of the station (e.g. type of device, position, ...). We recommend you copy-paste an existing one to get the correct structure.

.. note::
    The **issue.yml** files are optional (and if missing are initialized to be empty). These files allow the reader to skip the loading of the data according to time-periods (for example, due to temporal device failures). `Step 2 <#step-2-read-and-analyse-the-data>`_ will guide you through the analysis of your data in order to possibly found (and remove) these errors.



Data processed folder
======================

| 📁 DISDRODB/
| ├── 📁 Processed/
|    ├── 📁 NAME_OF_INSTITUTION_OR_COUNTRY/
|       ├── 📁 NAME_OF_CAMPAIGN/
|           ├── 📁 L0A
|               ├── 📁 <ID of the station>/
|                  ├── 📜 \*.paquet
|           ├── 📁 L0B
|               ├── 📁 ID of the station/
|                  ├── 📜 \*.paquet
|           ├── 📁 info
|           ├── 📁 metadata
|               ├── 📜 <ID of the station>.yml




Step 2 : Read and analyse the data
******************************************************************

Once the data structure is ready, you can start analyzing its content. To do so, we provide you with a bunch of tools gathered into ``disdrodb\L0\templates\reader_template.py``.

Copy and paste ``\templates\reader_template.py`` into  ``\templates\<Name of the institution or country>\temp_parser_<Name ot the campaign>.py`` and start digging into your data.

.. note::
	**Why do we need temp_parser_<NAME_OF_CAMPAIGN>.py ?**
	This file is designed to help the creation of a new reader.
	The input raw structure and content can be very different from one measurement to another.
	Therefore, we have to uniform it in order to match the common data model. ``temp_parser_<NAME_OF_CAMPAIGN>.py`` give us some tools to parameterize and modify and visualize the initial raw file.



In this file, you must first define some parameters (e.g. path of your data, loading parameters). Once the row data is loaded, you can comment and uncomment print functions to be sure your data is correctly shaped.

Relevent elements :

* The ``reader_kwargs`` dictionary that guides panda / dask reading
* The ``column_names`` list that defines the raw column names (according to the output model, see ``disdrodb\L0\configs\<type of device>\L0A_encodings.yml``)
* The ``df_sanitizer_fun()`` function that defines the processes to apply on the dataframe in order for the data to match the output data model.

Once your are happy with the state of your data, all these elements can be tranfered into the reader in `Step 3 <#step-3-create-the-reader>`_ .




Step 3 : Create the reader
******************************************************************

In this final step, the new reader is created and will be published to the community. It is therefore important to follow the initial file structure in order to guaranty consistencies between readers.

To do so, copy and paste ``disdrodb\L0\readers\parser_template.py`` into  ``\readers\<Name of the institution or country>\parser_<Name ot the campaign>.py`` and start digging  into it.

The relevant elements that have been defined  in `Step 2 <#step-2-read-and-analyse-the-data>`_  must be retranscripted here.

Once ready, `the reader can be run <#running-a-reader>`_ .



Adding a new reader : Tutorial
################################################

Please visit the following page to access to the tutorial notebook : 


.. toctree::
   :maxdepth: 1

   data_analysis

If you want to access to this notebook interactivally, you have to run jupyter notebook as follow : 

1. Go to the disdrodb folder

2. Start jupyter notebook 

	.. code-block:: bash

		python -m notebook

	
	*Make sure to be within your project virtual environment or conda environment.* 
	
3. Navigate to ``disdrodb\L0\templates\data_analysis.ipynb``




Adding a new reader : Tutorial (old - to be deleted )
################################################

In this tutorial, we will create a new reader based on a lightweight data sample.

This tutorial is divided into 3 parts :

* Step 1 : `Data <#tutorial-data>`_, where  we introduce the sample data.
* Step 2 : `Step 2 <#tutorial-step-2-read-and-analyse-the-data>`_ where we dig into the data to set up the transformation parameters.
* Step 3 : `Step 3 <#tutorial-step-3-create-the-reader>`_ , where we create the reader




Tutorial Data
******************************************************************

You will find the sample data for this tutorial in the folder``data`` of the GitHub repository.
It corresponds to one measurement campaign composed of two stations (``ID_station_1`` and ``ID_station_2``) during two days.

| 📁 data/
| 	📁 DISDRODB/
| 	├── 📁 Raw/
|    	├── 📁 INSTITUTION_or_COUNTRY/
|       	├── 📁 CAMPAIGN/
|           	├── 📁 data
|               	├── 📁 ID_station_1/
|                  	├── 📜 file60_20180817.dat.gz
|                  	├── 📜 file60_20180818.dat.gz
|               	├── 📁 ID_station_2/
|                  	├── 📜 file61_20180817.dat.gz
|                  	├── 📜 file61_20180818.dat.gz
|           	├── 📁 info
|           	├── 📁 issue
|               	├── 📜 ID_station_1.yml
|               	├── 📜 ID_station_2.yml
|           	├── 📁 metedata
|               	├── 📜 ID_station_1.yml
|               	├── 📜 ID_station_2.yml

This structure fulfills the requirements described  `here <#step-1-set-the-data-folder-for-raw-and-processed-datasets>`_ .

Tutorial step 2: Read and analyse the data
******************************************************************


**Objective** : Define the reading parameters.

**Folder** : ``disdrodb\L0\templates\TUTORIAL``


To read and analyse the data, we will use the file ``disdrodb\L0\templates\reader_template.py``. This file will help us to set the loading parameters correctly, and allow us to analyse the data.

#. First thing to do, copy and paste ``disdrodb\L0\templates\reader_template.py`` into ``disdrodb\L0\templates\TUTORIAL``

#. Then rename the copied file ``temp_parser_TUTORIAL.py``

#. Add the root folder to the path variable

	If you are running the code from DISDRODB project root folder, you need to add :


	.. code-block::
		:caption: Add these lines after the other import statements

		import sys
		sys.path.insert(0,os.getcwd())

	If your are running the script from the ``disdrodb\L0\templates`` folder, you don't have to do that.

#. Adapt the input and output paths

	This file is composed by 9 sections delimited by headers like :

	.. code-block::

		######################################
		#### 1. Define campaign filepaths ####
		######################################

	In the first section, we need to fill in input and output paths

	.. code-block::
		:caption: Before

		raw_dir = "<local_path>"  # Must end with campaign_name upper case
		processed_dir = "<local_path>"  # Must end with campaign_name upper case


	.. code-block::
		:caption: After

		raw_dir = os.path.join(os.getcwd(),"data/DISDRODB/Raw/INSTITUTION_or_COUNTRY/CAMPAIGN")
		processed_dir = os.path.join(os.getcwd(),"data/DISDRODB/Processed/INSTITUTION_or_COUNTRY/CAMPAIGN")

	.. note::
		These paths depend on where you are running your python script. In the tutorial, it is run from the project root folder. But feel free to adapt the paths.


#. If desired, change the station ID

	.. code-block::

		######################################################
		#### 3. Select the station for reader development ####
		######################################################


	Optional - In section 3, you may change the station used in parser development, using its ID correponding the the (n-1)th station in the directory (listed in alphabetical order).

	.. code-block::
		:caption: Example

		station_id = list_stations_id[1]


#. Change the file format

	.. code-block::

		##########################################################################
		#### 4. List files to process  [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
		##########################################################################


	In section 4, we define the file format (e.g., CSV). The default setting is ok for us since we have ``.dat`` files,
	 so we don't need to change anything here.

	.. code-block::

		glob_pattern = os.path.join("data", station_id, "*.dat*")  # CUSTOMIZE THIS


#. Set the dataframe reading properties

	.. code-block::

		#########################################################################
		#### 5. Define reader options [TO CUSTOMIZE AND THEN MOVE TO PARSER] ####
		#########################################################################


	In the ``reader_kwargs`` dictionary, you may set `any arguments <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_ that need to be passed
	for the reading of the raw data to a dataframe via Pandas.

	In our case, our raw file does not have a header on the first row.
	Therefore, we just need to add the following elements to ``reader_kwargs`` :

	.. code-block::
		:caption: Add this line

		reader_kwargs['header'] = None

#. Data exploration

	.. code-block::

		####################################################
		#### 6. Open a single file and explore the data ####
		####################################################

	The settings for the loading of the data is now ready, we can now load one file and analyse its content to see if there is any errors or inconsistencies.

	The following functions help us to get information about the content and the schema of the data. They can be commented or uncommented :

	*	``print_df_first_n_rows()`` : to print first rows.
	* 	``print_df_columns_unique_values()`` : to print unique values (one or many columns).
	* 	``infer_df_str_column_names()`` : Try to guess the column name based on string patterns (*according to L0A_encodings.yml and the type of sensor.*)
	*   ``print_valid_L0_column_names()`` : Print the valid column names (*according to L0A_encodings.yml and the type of sensor.*)


#. Define the columns names

	.. code-block::

		######################################################################
		#### 7. Define dataframe columns [TO CUSTOMIZE AND MOVE TO PARSER] ###
		######################################################################


	The data structure and content have been analyzed in detail during the previous steps. It is now time to formalize the columns names.


	.. code-block::
		:caption: Before

		column_names = [
			"id",
			"latitude",
			"longitude",
			"time",
			"datalogger_temperature",
			"datalogger_voltage",
			"rainfall_rate_32bit",
			"rainfall_accumulated_32bit",
			"weather_code_synop_4680",
			"weather_code_synop_4677",
			"reflectivity_16bit",
			"mor_visibility",
			"laser_amplitude",
			"number_particles",
			"sensor_temperature",
			"sensor_heating_current",
			"sensor_battery_voltage",
			"sensor_status",
			"rainfall_amount_absolute_32bit",
			"error_code",
			"raw_drop_concentration",
			"raw_drop_average_velocity",
			"raw_drop_number",
			"datalogger_error",
		]



	.. code-block::
		:caption: After

		column_names = [
			"id",
			"latitude",
			"longitude",
			"time",
			"datalogger_temperature",
			"datalogger_voltage",
			"rainfall_rate_32bit",
			"rainfall_accumulated_32bit",
			"weather_code_synop_4680",
			"weather_code_synop_4677",
			"reflectivity_32bit",
			"mor_visibility",
			"laser_amplitude",
			"number_particles",
			"sensor_temperature",
			"sensor_heating_current",
			"sensor_battery_voltage",
			"sensor_status",
			"rainfall_amount_absolute_32bit",
			"error_code",
			"raw_drop_concentration",
			"raw_drop_average_velocity",
			"raw_drop_number",
			"datalogger_error",
		]

	.. note::
		You may notice that the latitude and longitude are not real coordinates in this dataset. This is not important at this point since the column will be removed later and the real coordinates will be taken from the station’s metadata.


	We have now a couple of functions that help us to analyse the loaded dataset

	*	``check_column_names()`` : Checks that the column names respects DISDRODB standards *(according to L0A_encodings.yml and the type of sensor)*.
	*	``print_df_column_names()`` : Print dataframe columns names.
	*	``print_df_random_n_rows()`` : Print the content of the dataframe by column.
	*	``print_df_summary_stats()`` : Print some statistics.
	*	``print_df_columns_unique_values()`` :  Print unique values (one or many columns).
	*	``get_df_columns_unique_values_dict()`` :  Get a dictionary of the column names and unique value, respectively as key and value.



#. Process the content of the dataframe

	.. code-block::

		#########################################################
		#### 8. Implement ad-hoc processing of the dataframe ####
		#########################################################

	Now we have to drop the unrequired columns for L0

	.. code-block::
		:caption: Before

		df = df.drop(columns=["id", "latitude", "longitude"])


	.. code-block::
		:caption: After

		df = df.drop(columns=["id", "latitude", "longitude","datalogger_error",'datalogger_voltage','datalogger_temperature'])


	.. code-block::
		:caption: Before

		print_df_columns_unique_values(df, column_indices=slice(0, 20), column_names=True)


	.. code-block::
		:caption: After

		print_df_columns_unique_values(df, column_indices=slice(0, 17), column_names=True)




#. Simulate parser file code execution

	.. code-block::

		################################################
		#### 9. Simulate parser file code execution ####
		################################################

	Now we have to modify the ``df_sanitizer_fun()`` function. This function will be used as an argument into the  ``read_L0A_raw_file_list()``.

	.. code-block::
		:caption: Before

		columns_to_drop = [
			"id",
		]


	.. code-block::
		:caption: After

		columns_to_drop = [
			"id",
			"datalogger_temperature",
			"datalogger_voltage",
			"datalogger_error",
		]


	.. code-block::
		:caption: Before

		print_df_columns_unique_values(df, column_indices=slice(0, 20), column_names=True)


	.. code-block::
		:caption: After

		print_df_columns_unique_values(df, column_indices=slice(0, 17), column_names=True)



Congratulation, Part 1 of this tutorial is done ! If the data printed is correct, we can move on to create the proper reader !


Tutorial step 3 : Create the reader
******************************************************************


**Objective** : Transcribe the reading parameters into a proper reader.

**Folder** : ``disdrodb\L0\readers\TUTORIAL``

We have now all the element to start creating the new reader. All the modifications that we did in the file ``temp_parser_TUTORIAL.py`` must be now transcribed into a reader file.

#. Copy and paste the ``disdrodb\L0\readers\parser_template.py`` into the folder ``disdrodb\L0\readers\TUTORIAL``

#. Rename the copied file ``parser_TUTORIAL.py``



#. Add the root folder to the path variable

	.. code-block::
		:caption: Before

		import click
		from disdrodb.L0 import run_L0

	.. code-block::
		:caption: After

		import os
		import sys
		sys.path.insert(0,os.getcwd())
		import click
		from disdrodb.L0.L0_processing import run_L0

#. Define the columns names

	.. code-block::
		:caption: Before

		column_names = []

	.. code-block::
		:caption: After

		column_names = [
			"id",
			"latitude",
			"longitude",
			"time",
			"datalogger_temperature",
			"datalogger_voltage",
			"rainfall_rate_32bit",
			"rainfall_accumulated_32bit",
			"weather_code_synop_4680",
			"weather_code_synop_4677",
			"reflectivity_32bit",
			"mor_visibility",
			"laser_amplitude",
			"number_particles",
			"sensor_temperature",
			"sensor_heating_current",
			"sensor_battery_voltage",
			"sensor_status",
			"rainfall_amount_absolute_32bit",
			"error_code",
			"raw_drop_concentration",
			"raw_drop_average_velocity",
			"raw_drop_number",
			"datalogger_error",
		]

#. Add raw data loading parameter

	.. code-block::
		:caption: Before

		reader_kwargs["blocksize"] = None # "50MB"


	.. code-block::
		:caption: After

		reader_kwargs["blocksize"] = None # "50MB"
		reader_kwargs['header'] = None

#. Modify the ``df_sanitizer_fun()`` function

	.. code-block::
		:caption: Before

		def df_sanitizer_fun(df, lazy=False):
			# Import dask or pandas
			if lazy:
				import dask.dataframe as dd
			else:
				import pandas as dd

			# - Drop datalogger columns
			columns_to_drop = ['id', 'datalogger_temperature', 'datalogger_voltage', 'datalogger_error']
			df = df.drop(columns=columns_to_drop)

			# - Drop latitude and longitude
			# --> Latitude and longitude is specified in the the metadata.yaml
			df = df.drop(columns=['latitude', 'longitude'])

			# - Convert time column to datetime with resolution in seconds
			df['time'] = dd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')

			return df


	.. code-block::
		:caption: After

		def df_sanitizer_fun(df, lazy=False):
			# Import dask or pandas
			if lazy:
				import dask.dataframe as dd
			else:
				import pandas as dd

			# - Drop datalogger columns
			columns_to_drop = [
			"id",
			"datalogger_temperature",
			"datalogger_voltage",
			"datalogger_error"
			]

			df = df.drop(columns=columns_to_drop)

			# - Drop latitude and longitude
			# --> Latitude and longitude is specified in the the metadata.yaml
			df = df.drop(columns=['latitude', 'longitude'])

			# - Convert time column to datetime with resolution in seconds
			df['time'] = dd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')

			return df


#. Run the script

	From the root folder, just run :

	.. code-block::


		 python .\disdrodb\L0\readers\TUTORIAL\parser_TUTORIAL_finished.py  <data_folder>\DISDRODB\Raw\INSTITUTION_or_COUNTRY\CAMPAIGN\ <data_folder>\DISDRODB\Processed\INSTITUTION_or_COUNTRY\CAMPAIGN\ -l0b True -f True -v True -d False


	You need to adapt the <data_folder> parameter to your local data folder.

	Have a look  `here <#runing-a-reader>`_ if you want to customize this command.


#. Check if the script has correctly run

	The output folder should be as follow :

	| 📁 DISDRODB/
	| ├── 📁 Processed/
	|    ├── 📁 INSTITUTION_or_COUNTRY/
	|       ├── 📁 CAMPAIGN/
	|           ├── 📁 info
	|               ├── 📜 ID_station_1.yml
	|               ├── 📜 ID_station_2.yml
	|           ├── 📁 L0A
	|               ├── 📁 ID_station_1/
	|                  ├── 📜 _sID_station_1.parquet
	|               ├── 📁 ID_station_2
	|                  ├── 📜 _sID_station_2.parquet
	|           ├── 📁 L0B
	|               ├── 📁 ID_station_1/
	|                  ├── 📜 _sID_station_1.nc
	|               ├── 📁 ID_station_2/
	|                  ├── 📜 _sID_station_2.nc
	|           ├── 📁 logs
	|               ├── 📜 <date>_LO_parser.log
	|           ├── 📁 metadata
	|               ├── 📜 ID_station_1.yml
	|               ├── 📜 ID_station_2.yml


Well done 👋  you have created a new reader. You can now :

* Create you own reader based on your data.
* Run this reader over your full dataset to generate L0 files.
* Publish this reader to the github main repository to enrich the DISDRODB project ! Have a look at the `contributors guidelines <https://disdrodb.readthedocs.io/en/latest/contributors_guidelines.html>`_



.. note::
    Corrections of this tutorial can be found here :

	* Part 1 : ``\disdrodb\L0\templates\TUTORIAL\temp_parser_TUTORIAL_correction.py``
	* Part 2 : ``disdrodb\L0\readers\TUTORIAL\parser_TUTORIAL_correction.py``


