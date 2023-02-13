3:=========================
Readers
=========================


DISDRODB supports reading and loading data from many input file formats.
The following subsections describe, first, what a reader is and how it can be defined.
Then, it illustrates multiple methods how a reader can be called (i.e. from terminal or within python)
to process raw data into DISDRODB L0 products.

What is a reader
======================

A DISDRODB reader is python function encoding all the required information to convert 
raw disdrometer text (or netcdf) data into DISDRODB L0A and/or DISDRODB L0B products. 

To be more precise, a reader contains:
1. a glob string specifying the pattern to select all files to process within a station directory
2. the name of the variables present in the raw files (i.e. the file header/columns) 
3. some special arguments required to open and read the raw files (i.e the delimiter)
4. an optional ad-hoc function to make the raw data compliant with the DISDRODB standards.

If the raw data are text-based files, the reader will take care of first converting the data 
into the DISDRODB L0A dataframe format, and subsequently to reshape the data into the DISDRODB L0B netCDF format.
Instead, if the raw data are netCDFs files, the reader will take care to the reformat the source netCDF into 
the DISDRODB L0B netCDF format.

In the DISDRODB metadata of each station, the ``reader`` key specifies the DISDRODB reader required to
to process the raw data.
This enable to process the DISDRODB archive 


Available readers
======================

The readers are archived in the disdrodb software by data source. 
You can have a preliminary look on how the readers looks like by exploring 
the `DISDRODB.L0.readers directory <https://github.com/ltelab/disdrodb/tree/main/disdrodb/L0/readers>`_

The function `available_readers` returns a dictionary with all readers currently available within DISDRODB`.
By specifying the ``data_sources`` argument, only the readers for the specified data sources are returned.

.. code-block:: python

	from disdrodb.L0 import available_readers
	available_readers()
	available_readers(data_sources=["EPFL", "GPM"])

The dictionary has the following shape: 

.. code-block::

	`{"<DataSource1>": [<ReaderName1>, <ReaderName2>],
		...
	  "<DataSourceN": [<ReaderNameY>, <ReaderNameZ>]
	}`


Reader structure   
======================

A reader it s defined by the following arguments:

.. code-block:: python

    def reader(
        raw_dir,
        processed_dir,
        station_name,
        # Processing options
        force=False,
        verbose=False,
        parallel=False,
        debugging_mode=False,
    ):
    

* ``raw_dir`` : str - Directory path where all the raw data of a specific campaign/network are stored.

		* The path must have the following structure: '<...>/DISDRODB/Raw/<data_source>/<campaign_name>'.
		* Inside the raw_dir directory, the software expects to find the following structure:
           * <raw_dir>/data/<station_name>/<raw_files>
           * <raw_dir>/metadata/<station_name>.yaml


* ``processed_dir`` : str - Desired directory path for the processed DISDRODB L0A and L0B products.

        * The path should have the following structure: '<...>/DISDRODB/Processed/<data_source>/<campaign_name>'
        * The <campaign_name> must match with the one specified in the raw_dir path.
        * For reader testing purposes, you can define i.e. '/tmp/DISDRODB/Processed/<data_source>/<campaign_name>'
   
   
* ``station_name`` : str - Name of the station to be processed. 

		
* ``--force`` : bool [true\| **false** ] - Whether to overwrite existing data.

        *  If True, overwrite existing data into destination directories.
        *  If False, raise an error if there are already data into destination directories.


* ``--verbose`` : bool [true\| **false** ] - Whether to print detailed processing information into terminal.


* ``--debugging_mode`` : bool [true\| **false** ] -  If True, it reduces the amount of data to process.

        * It processes just 3 raw data files.

* ``--parallel`` : bool [ **true** \|false] - Whether to process multiple files simultanously.

        * If parallel=False, the raw files are processed sequentially. 
        * If parallel=True, each file is processed in a separate core.  


Inside the reader function, a few components must be customized. 


Reader components for raw text files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

If the input raw data are text files, the reader must defines the following components:

1. The ``glob_patterns`` to search for the raw data files within the <raw_dir>/data/<station_name> directory.

2. The ``column_names`` list defines the header of the raw text file.

3. The ``reader_kwargs`` dictionary containing all specifications to open the text file into 
   a pandas dataframe.
   The  possible key-value arguments are listed `here <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_

4. The ``df_sanitizer_fun(df)`` function takes as input the raw dataframe and apply ad-hoc
   processing to make the dataframe compliant to the DISDRODB L0A standards. 
   Typically, this function is used to drop columns not compliant with the expected set of DISDRODB variables
   and to create the DISDRODB expected ``time`` column into UTC datetime format.
   In the output dataframe, each row must correspond to a timestep!

It's important to note that the internal L0A processing already takes care of: 
* removing rows with undefined timestep
* removing rows with corrupted values
* sanitize string column with trailing spaces 
* dropping rows with duplicated timesteps (keeping only the first occurence)

In DISDRODB L0A format, the raw precipitation spectrum, named ``raw_drop_number`` , it is expected
to be defined as a string with a series of values seperated by a delimiter like ``,`` or ``;``. 
Therefore, the ``raw_drop_number`` field value is expected to look like ``"000,001,002, ..., 001"``
For example, if the ``raw_drop_number`` looks like the following three cases, you need to preprocess it accordingly 
into the ``df_sanitizer_fun``: 

* Case 1: ``"000001002 ...001"``. Convert to ``"000,001,002, ..., 001"``.  Example reader `here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/L0/readers/NETHERLANDS/DELFT.py>`_ 
* Case 2: ``"000 001 002 ... 001"``. Convert to ``"000,001,002, ..., 001"``.  Example reader `here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/L0/readers/CHINA/CHONGQING.py>`_
* Case 3: ``",,,1,2,...,,,"``. Convert to ``"0,0,0,1,2,...,0,0,0"``.  Example reader here

Finally, the reader will call the ``run_l0`` function, by passing to it all the above described arguments. 

.. code-block:: python
    run_l0a(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            station_name=station_name,
            # Custom arguments of the reader for L0A processing
            glob_patterns=glob_patterns,
            column_names=column_names,
            reader_kwargs=reader_kwargs,
            df_sanitizer_fun=df_sanitizer_fun,
            # Processing options
            force=force,
            verbose=verbose,
            parallel=parallel,
            debugging_mode=debugging_mode,
        )


Reader components for raw netCDF files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

On the other hand, if the input raw data are netCDF files, the reader must defines the following components: 

1. The ``glob_patterns`` to search for the raw netCDF files within the <raw_dir>/data/<station_name> directory.

2. The ``dict_names`` dictionary mapping the dimension and variables names of the source netCDF to the DISDRODB L0B standards.
   Variables not present the dict_names are dropped from the dataset.
   Variables specified in dict_names but missing in the dataset, are added as NaN arrays.
   Here is an example of dict_names: 
   .. code-block:: python
        dict_names = {
            # Dimensions 
            "timestep": "time"
            "diameter_bin": "diameter_bin_center"
            "velocity_bin": "velocity_bin_center"
            # Variables
            "reflectivity": "reflectivity_32bit"
            "precipitation_spectrum": "raw_drop_number",
        }

3. The ``ds_sanitizer_fun(ds)`` function takes as input the raw netCDF file (in xr.Dataset format) and apply ad-hoc
   processing to make the xr.Dataset compliant to the DISDRODB L0B standards.
   Typically, this function is used to drop xr.Dataset coordinates not compliant with the expected set of DISDRODB coordinates.


Finally, the reader will call the ``run_l0b_from_nc`` function, by passing to it all the above described arguments. 
 
.. code-block:: python
  run_l0b_from_nc(
      raw_dir=raw_dir,
      processed_dir=processed_dir,
      station_name=station_name,
      # Custom arguments of the reader
      glob_patterns=glob_patterns,
      dict_names=dict_names,
      ds_sanitizer_fun=ds_sanitizer_fun,
      # Processing options
      force=force,
      verbose=verbose,
      parallel=parallel,
      debugging_mode=debugging_mode,
  )


Adding a new reader
======================

We describe here the 4 steps to create a reader locally. 
To share the reader with the community, please also read the `Contributing guide <contributors_guidelines.html>`__.


* `Step 1 <#step-1-set-the-folder-structure-for-raw-and-processed-datasets>`_ : Set the DISDRODB "Raw" directory structure
* `Step 2 <#step-2-analyse-the-data-and-define-the-reader-components>`_ :  Analyse the data and implement the reader
* `Step 3 <#step-3-create-and-share-your-reader>`_ :  Share the reader
* `Step 4 <#step-4-define-reader-testing-files>`_ :  Create the test files
* TODO Description
* Step X: add metadata and check validity 
* Step X: check the reader is searchable 
* Step X: check the L0 processing with run_disdrodb_l0_station 



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
|                    ├── 📜 \*.\*  : raw files
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



The "Processed Directories Tree" will be created automatically when launching the DISDRODB L0 processing.
It will look like this:

| 📁 DISDRODB
| ├── 📁 Processed
|    ├── 📁 `<data_source>`
|       ├── 📁 `<campaign_name>`
|           ├── 📁 L0A
|               ├── 📁 `<station_name>`
|                   ├── 📜 \*.parquet
|           ├── 📁 L0B
|               ├── 📁 `<station_name>`
|                    ├── 📜 \*.nc
|           ├── 📁 info
|           ├── 📁 metadata
|               ├── 📜 <station_name>.yml


Step 2 : Analyse the data and define the reader components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the data structure is ready, you can start analyzing its content. 
To do so, we provide you with a jupyter notebook at ``disdrodb\L0\readers\reader_preparation.ipynb`` that should facilitate the task.
We highly suggest to copy the notebook and adapt it to your own data.

In this notebook, we guide you through the definition of 4 relevant DISDRODB reader components: 

* The ``glob_patterns`` to search for the data files within the ``.../data/<station_name>`` directory.
    
* The ``reader_kwargs`` dictionary guides the pandas / dask dataframe creation. 

For more information on the possible key-value arguments, read the `pandas <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_
documentation.
 
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


