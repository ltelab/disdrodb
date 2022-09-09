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

Once the data structure is ready, you can start analyzing its content. To do so, we provide you with tools in ``disdrodb\L0\readers\reader_preparation.ipynb``.
Copy the notebook and adapt it to your own data.

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

Please visit the following page to access a read only tutorial notebook :


.. toctree::
   :maxdepth: 1

   data_analysis

If you want interactive notebook, you need to run jupyter notebook in your local machine. Proceed as follow :

1. Make sure you have the latest version of the code in your local folder. See the git clone command in the `Installation for developers <https://disdrodb.readthedocs.io/en/latest/install.html#installation-for-developers>`_ section.

2. Enter your project virtual environment or conda environment.  Please, refer to the `Installation for developers <https://disdrodb.readthedocs.io/en/latest/install.html#installation-for-developers>`_ section if needed.

3. Navigate to the disdrodb folder

4. Start jupyter notebook

	.. code-block:: bash

		python -m notebook

	It starts your default web browser with jupyter notebook main page.


5. Navigate to ``disdrodb\L0\readers`` and double click on `reader_preparation.ipynb`.

6. You can now start using the tutorial notebook.






