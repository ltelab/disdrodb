=========================
Readers
=========================



DISDRODB supports reading and loading data from many input file formats and schemes. The following sections describe the different way data can be loaded, requested, or added to the DISDRODB project.



Available Readers
======================

The following function returns the dictionary of all readers `.

.. code-block:: python

	from disdrodb.L0.L0_processing import get_available_readers
	get_available_readers()


The resulting dictionary has the following shape: 


.. code-block::

	`{
		"Data source 1": 
			{
				"Campaign name 1 ":"File path 1" ,
				"Campaign name 2 ":"File path 2" ,
				...
				"Campaign name n ":"File path n" ,
			
			}
		...
		"Data source n": 
			{
				"Campaign name 1 ":"File path 1" ,
				"Campaign name 2 ":"File path 2" ,
				...
				"Campaign name n ":"File path n" ,
			
			}
	}`





Using a reader
======================

Running a reader can be done by command line or directly in python. In both ways, the following parameters must or could be defined. 




Readers parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ``data_source`` : str - Name of the data source. 

		* Example data_source: 'EPFL'.
		* Check the `available Readers <#available-readers>`__ function to get the list of the available data sources. 

* ``campaign_name`` : str - Name of the campaign. 

		* Example data_source: 'EPFL_ROOF_2012'.
		* Check the `available Readers <#available-readers>`__  function to get the list of the available campaign.  

* ``raw_dir`` : str - Directory path of raw file for a specific campaign.

		* The path should end with <campaign_name>.
		* Example raw_dir: '<...>/disdrodb/data/raw/<campaign_name>'.


* ``processed_dir`` : str - Desired directory path for the processed L0A and L0B products.

		* The path should end with <campaign_name> and match the end of raw_dir.
		* Example: '<...>/disdrodb/data/processed/<campaign_name>'.

* ``--l0a_processing`` : bool [ **true** \|false] - Whether to launch processing to generate DISDRODB L0A Apache Parquet file(s) from raw data.




* ``--l0b_processing`` : bool [ **true** \|false] - Whether to launch processing to generate DISDRODB L0B netCDF4 file(s) from L0A data.




* ``--keep_l0a`` : bool [true\| **false** ] - Whether to keep the L0A files after having generated the L0B netCDF products.



* ``--force`` : bool [true\| **false** ] - Whether to overwrite existing data.

*  If True, overwrite existing data into destination directories.
*  If False, raise an error if there are already data into destination directories.


* ``--verbose`` : bool [true\| **false** ] -  Whether to print detailed processing information into terminal.



* ``--debugging_mode`` : bool [true\| **false** ] -  If True, it reduces the amount of data to process.

* For L0A processing, it processes just 3 raw data files.
* For L0B processing, it takes a small subset of the L0A Apache Parquet dataframe.



* ``--lazy`` : bool [ **true** \|false] - Whether to perform processing lazily with dask.

* If lazy=True, it employs dask.dataframe and dask.array. 

* If lazy=False, it employs pandas.DataFrame and numpy.array.




* ``--single_netcdf`` : bool  [ **true** \| false] - Whether to concatenate all raw files into a single DISDRODB L0B netCDF file.


* If single_netcdf=True, all raw files will be saved into a single L0B netCDF file.
* If single_netcdf=False, each raw file will be converted into the corresponding L0B netCDF file.


Running a reader
~~~~~~~~~~~~~~~~~~~~~~~~


There are two ways of running a reader. 

1. By command line : 


	.. code-block::

		run_disdrodb_l0_reader data_source campaign_name raw_dir processed_dir [parameters]

	
	Where the parameters are defined `here <#readers-parameters>`__.


	Example :

	.. code-block::

		run_disdrodb_l0_reader NETHERLANDS DELFT "...\DISDRODB\Raw\NETHERLANDS\DELFT" "...\DISDRODB\Processed\NETHERLANDS\DELFT" --l0a_processing True --l0b_processing False --keep_l0a True --force True --verbose True --debugging_mode False --lazy False --single_netcdf False 
	 


2. By calling a python function 

	2.1 Wrapping function : 

		.. code-block:: python

			from disdrodb.L0.L0_processing import run_reader
			run_reader(<data_source>, <campaign_name>, <raw_dir>, <processed_dir>, ...)

	
		Example :

		.. code-block:: python

			from disdrodb.L0.L0_processing import run_reader

			raw_dir = "...\\DISDRODB\\Raw\\NETHERLANDS\\DELFT"
			processed_dir = "...\\DISDRODB\\Processed\\NETHERLANDS\\DELFT"
			l0a_processing=True
			l0b_processing=True
			keep_l0a=True
			force=True
			verbose=True
			debugging_mode=True
			lazy=True
			single_netcdf=True

			run_reader('NETHERLANDS','DELFT',raw_dir,processed_dir,l0a_processing,l0b_processing,keep_l0a,force,verbose,debugging_mode,lazy,single_netcdf)

	
	2.2 From the reader itself : 

		.. code-block:: python

			from disdrodb.L0.readers.NETHERLANDS.DELFT import reader

			raw_dir = "...\\DISDRODB\\Raw\\NETHERLANDS\\DELFT"
			processed_dir = "...\\DISDRODB\\Processed\\NETHERLANDS\\DELFT"
			l0a_processing=True
			l0b_processing=True
			keep_l0a=True
			force=True
			verbose=True
			debugging_mode=True
			lazy=True
			single_netcdf=True

			reader(raw_dir,processed_dir,l0a_processing,l0b_processing,keep_l0a,force,verbose,debugging_mode,lazy,single_netcdf)



|



Adding a new reader
======================

We describe here the 4 steps to create a reader locally. To publish the reader to the community, please refer to the `Contributing guide <contributors_guidelines.html>`__.


* `Step 1 <#step-1-set-the-folder-structure-for-raw-and-processed-datasets>`_ : Set the folder structure for raw and processed datasets
* `Step 2 <#step-2-analyse-the-data-and-define-the-reader-components>`_ :  Read and analyse the data
* `Step 3 <#step-3-create-and-share-your-reader>`_ :  Create the reader
* `Step 4 <#step-4-define-reader-testing-files>`_ :  Create the test files




See also the step-by-step `tutorial <#adding-a-new-reader-tutorial>`_   that will demonstrate in detail all these steps with a sample lightweight dataset.


Step 1 : Set the folder structure for raw and processed datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The raw and processed data folder must follow strictly the following structure:

*Raw data folder* :


| 📁 DISDRODB
| ├── 📁 Raw
|    ├── 📁 `<data_source>`
|       ├── 📁 `<campaign_name>`
|           ├── 📁 data
|               ├── 📁 `<station_name>`
|                  ├── 📜 \*.\*  : raw files
|           ├── 📁 info
|           ├── 📁 issue
|               ├── 📜 <station_name>.yml
|           ├── 📁 metadata
|               ├── 📜 <station_name>.yml


.. note::
	Guidelines for the naming of the **<data_source>** directory :

	* We use the institution name when campaign data spans more than 1 country.
	* We use country when all campaigns (or sensor networks) are inside a given country.

.. note::
    For each folder in the `/data` directory (for each station) there must be an equally named **\*.yml** file in the `/metadata` folder. 
    The **metadata YAML** file contains relevant information of the station (e.g. type of device, position, ...) which are required for the correct processing and integration into the DISDRODB database.
    We recommend you to copy-paste an existing metadata YAML file to get the correct structure.
    
    .. warning::
    	TODO: Add section explaining all the metadata keys 
	
    	TODO: Add an empty metadata yaml somewhere in the repo and link to it! 

.. note::
    The **issue YAML** files are optional (and if missing are initialized to be empty).
    These files allow the reader to skip the loading of the data according to time-periods (for example, due to temporal device failures).
    `Step 2 <#step-2-analyse-the-data-and-define-the-reader-components>`_ will guide you through the analysis of your data in order to possibly found (and remove) these errors.



*Data processed folder*  (for your information) :


| 📁 DISDRODB
| ├── 📁 Processed
|    ├── 📁 `<data_source>`
|       ├── 📁 `<campaign_name>`
|           ├── 📁 L0A
|               ├── 📁 `<station_name>`
|                  ├── 📜 \*.parquet
|           ├── 📁 L0B
|               ├── 📁 `<station_name>`
|                  ├── 📜 \*.parquet
|           ├── 📁 info
|           ├── 📁 metadata
|               ├── 📜 <station_name>.yml



Note that this folder will be created automatically, no need to create it while developping the new reader



Step 2 : Analyse the data and define the reader components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the data structure is ready, you can start analyzing its content. 
To do so, we provide you with a jupyter notebook at ``disdrodb\L0\readers\reader_preparation.ipynb`` that should facilitate the task.
We highly suggest to copy the notebook and adapt it to your own data.

In this notebook, we guide you through the definition of 4 relevant DISDRODB reader components: 

* The ``files_glob_pattern`` to search for the data files within the ``.../data/<station_name>`` directory.
    
* The ``reader_kwargs`` dictionary guides the pandas / dask dataframe creation. 

For more information on the possible key-value arguments, read the `pandas <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_
and/or `dask <https://docs.dask.org/en/stable/generated/dask.dataframe.read_csv.html>`_  documentation.
 
* The ``column_names`` list defines the column names of the readed raw text file. 

* The ``df_sanitizer_fun()`` function that defines the processing to apply on the readed dataframe in order for the dataframe to match the DISDRODB   standards.

The dataframe which is returned by the ``df_sanitizer_fun`` must have only columns compliants with the DISDRODB standards ! 
 
When this 4 components are correctly defined, they can be transcripted into the `reader_template.py <https://github.com/ltelab/disdrodb/blob/main/disdrodb/L0/readers/reader_template.py>`_ file which is now almost ready to be shared with the community.


Step 3 : Create and share your reader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you arrived at this final step, it means that your reader is now almost ready to be shared with the community. 
However, in order to guarantee consistency between readers, it is very important to follow a specific nomenclature.


Therefore, rename your modified `reader_template.py <https://github.com/ltelab/disdrodb/blob/main/disdrodb/L0/readers/reader_template.py>`_ file as ``reader_<CAMPAIGN_NAME>.py`` and copy it into the directory ``\disdrodb\LO\readers\<DATA_SOURCE>\``.


Step 4 : Define reader testing files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you plan to create a new reader, you must provide a testing data sample composed of a tiny raw file and the expected results. 


All readers are tested as follow : 

 
.. image:: /static/reader_testing.png



.. note:: 
	The objective is to run every reader sequentially. Therefore, make sure to provide a very small test sample in order to limit the computing time.

	
	The size of the test sample must just be sufficient to guarantee the detection of errors due to code changes.  

	

	A typical test file is composed of 2 stations, with 2 days of measurements with a couple of rows each. 
 
 

The `GitHub readers testing resources <https://github.com/EPFL-ENAC/LTE-disdrodb-testing>`_ must have the following structure: 

 
| 📁 LTE-disdrodb-testing 
| ├── 📁 disdrodb
|    ├── 📁 L0
|       ├── 📁 readers
|           ├── 📁 <data_source>
|               ├── 📁 <campaign_name>
|                  ├── 📜 raw.zip
|           			├── 📁 data
|               			├── 📁 <station_name>
|                  				├── 📜 \*.\*  : raw files
|           			├── 📁 issue
|               			├── 📜 <station_name>.yml
|          				├── 📁 metadata
|               			├── 📜 <station_name>.yml
|                  ├── 📜 processed.zip
|           			├── 📁 info
|               			├── 📜 <station_name>.yml
|           			├── 📁 L0A
|               			├── 📁 `station_name>
|                  				├── 📜 \*.\parquet
|           			├── 📁 L0B
|               			├── 📁 <station_name>
|               				├── 📜 \*.\nc 
|          				├── 📁 metadata
|               			├── 📜 <station_name>.yml



.. warning:: 
	
	Naming convention :
	
	**<data_source>**  : Name of the data source.
	
	* We use the institution name when campaign data spans more than 1 country.
	* We use country when all campaigns (or sensor networks) are inside a given country.
	* Must be in capital letter. 
	* Must correspond to the name of the folder where the reader python file has been saved. 
	* Example : `EPFL` or `ITALY`
	
	
	**<campaign_name>**  : Name of the campaign.

	* Must be in capital letter. 
	* Must correspond to the name of the reader python file. 
	* Example : `LOCARNO2018` or `GID`




Process as follow to add a new test file :

1. Clone the `LTE-disdrodb-testing GitHub repository <https://github.com/EPFL-ENAC/LTE-disdrodb-testing>`_


	.. code-block:: bash

		git clone https://github.com/EPFL-ENAC/LTE-disdrodb-testing.git
		
2. Add your file according structure described above 
3. Commit and push your changes
4. Test your test with pytest (have a look `here <contributors_guidelines.html#running-test>`__)






Tutorial - Reader preparation step-by-step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please visit the following page to access a read-only tutorial notebook:


.. toctree::
   :maxdepth: 1

   reader_preparation

If you want to run an interactive notebook, you need to run jupyter notebook in your local machine. Proceed as follow :

1. Make sure you have the latest version of the code in your local folder. See the git clone command in the `Installation for developers <https://disdrodb.readthedocs.io/en/latest/install.html#installation-for-developers>`_ section.

2. Enter your project virtual environment or conda environment.  Please, refer to the `Installation for developers <https://disdrodb.readthedocs.io/en/latest/install.html#installation-for-developers>`_ section if needed.

3. Navigate to the disdrodb folder

4. Start jupyter notebook with 

	.. code-block:: bash

		python -m notebook
	
	or 
	
	.. code-block:: bash

		jupyter notebook

	This will open your default web browser with jupyter notebook on the main page.


5. Navigate to ``disdrodb\L0\readers`` and double click on the ``reader_preparation.ipynb``.

6. You can now start using the tutorial notebook.


