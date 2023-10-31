========
Readers
========


DISDRODB supports reading and loading data from many input file formats.
The following subsections describe, first, what a reader is and how it can be defined.
Then, it illustrates multiple methods how a reader can be called (i.e. from terminal or within python) to process raw data into DISDRODB L0 products.

What is a reader
=================

A DISDRODB reader is a python function encoding all the required information to convert
raw disdrometer text (or netCDF) data into DISDRODB L0A and/or DISDRODB L0B products.

To be more precise, a reader contains:

1. a glob string specifying the pattern to select all files to be processed within a station directory;

2. the name of the variables present in the raw files (i.e. the file header/columns);

3. some special arguments required to open and read the raw files (i.e the delimiter);

4. an optional ad-hoc function to make the raw data compliant with the DISDRODB standards.

If the raw data are text-based files, the reader will take care of first converting the data
into the DISDRODB L0A dataframe format, and subsequently to reshape the data into the DISDRODB L0B netCDF format.
Instead, if the raw data are netCDFs files, the reader will take care of reformatting the source netCDF into
the DISDRODB L0B netCDF format.

In the DISDRODB metadata of each station:

* the ``reader`` key specifies the DISDRODB reader required to process the raw data.

* the ``raw_data_format`` variable specifies whether the source data is in the form of txt or netcdf files.


Available readers
======================

In the in the disdrodb software, the readers are organized by data source.
You can have a preliminary look on how the readers looks like by exploring
the `DISDRODB.l0.readers directory <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers>`_

The function ``available_readers`` returns a dictionary with all readers currently available within DISDRODB.
By specifying the ``data_sources`` argument, only the readers for the specified data sources are returned.

.. code-block:: python

    from disdrodb.l0 import available_readers

    available_readers()
    available_readers(data_sources=["EPFL", "GPM"])

The dictionary has the following structure:

.. code-block:: text

    {
        "<DataSource1>": [<ReaderName1>, <ReaderName2>],
        ...
        "<DataSourceN>": [<ReaderNameY>, <ReaderNameZ>],
    }


Reader structure
======================

A reader is a function defined by the following input arguments:

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
        pass


* ``raw_dir`` : str - The directory path where all the raw data of a specific campaign/network are stored.

        * The path must have the following structure: ``<...>/DISDRODB/Raw/<data_source>/<campaign_name``.
        * Inside the raw_dir directory, the software expects to find the following structure:

            * ``<raw_dir>/data/<station_name>/<raw_files>``
            * ``<raw_dir>/metadata/<station_name>.yml``


* ``processed_dir`` : str - The desired directory path where to save the DISDRODB L0A and L0B products.

        * The path should have the following structure: ``<...>/DISDRODB/Processed/<data_source>/<campaign_name>``
        * The ``<campaign_name>`` must match with the one specified in the ``raw_dir``.
        * For reader testing purposes, you can define i.e. ``/tmp/DISDRODB/Processed/<data_source>/<campaign_name>``


* ``station_name`` : str - Name of the station to be processed.


* ``force`` : bool [true\| **false** ] - Whether to overwrite existing data.

        *  If ``True``, overwrite existing data into destination directories.
        *  If ``False``, raise an error if there are already data into destination directories.


* ``verbose`` : bool [true\| **false** ] - Whether to print detailed processing information into terminal.


* ``debugging_mode`` : bool [true\| **false** ] -  If ``True``, it reduces the amount of data to process.

        * It processes just 3 raw data files.

* ``parallel`` : bool [true\| **false** ] - Whether to process multiple files simultaneously.

        * If ``parallel=False``, the raw files are processed sequentially.
        * If ``parallel=True``, each file is processed in a separate core.


Inside the reader function, a few components must be customized.


Reader components for raw text files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the input raw data are text files, the reader must defines the following components:

1. The ``glob_patterns`` to search for the raw data files within the ``<raw_dir>/data/<station_name>`` directory.

2. The ``column_names`` list defines the header of the raw text file.

3. The ``reader_kwargs`` dictionary containing all specifications to open the text file into
   a pandas dataframe. The possible key-value arguments are listed `here <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_

4. The ``df_sanitizer_fun(df)`` function takes as input the raw dataframe and apply ad-hoc
   processing to make the dataframe compliant to the DISDRODB L0A standards.
   Typically, this function is used to drop columns not compliant with the expected set of DISDRODB variables
   and to create the DISDRODB expected ``time`` column into UTC datetime format.
   In the output dataframe, each row must correspond to a timestep !

It's important to note that the internal L0A processing already takes care of:

* removing rows with undefined timestep

* removing rows with corrupted values

* sanitize string column with trailing spaces

* dropping rows with duplicated timesteps (keeping only the first occurrence)

In the DISDRODB L0A format, the raw precipitation spectrum, named ``raw_drop_number`` ,
it is expected to be defined as a string with a series of values separated by a delimiter like ``,`` or ``;``.
Therefore, the ``raw_drop_number`` field value is expected to look like ``"000,001,002, ..., 001"``
For example, if the ``raw_drop_number`` looks like the following three cases, you need to preprocess it accordingly
into the ``df_sanitizer_fun``:

* Case 1: ``"000001002 ...001"``. Convert to ``"000,001,002, ..., 001"``.  Example reader `here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/NETHERLANDS/DELFT.py>`_
* Case 2: ``"000 001 002 ... 001"``. Convert to ``"000,001,002, ..., 001"``.  Example reader `here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/CHINA/CHONGQING.py>`_
* Case 3: ``",,,1,2,...,,,"``. Convert to ``"0,0,0,1,2,...,0,0,0"``.  Example reader `here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/FRANCE/SIRTA_OTT2.py>`_

Finally, the reader will call the ``run_l0a`` function, by passing to it all the above described arguments.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the other hand, if the input raw data are netCDF files, the reader must define the following components:

1. The ``glob_patterns`` to search for the raw netCDF files within the ``<raw_dir>/data/<station_name>`` directory.

2. The ``dict_names`` dictionary mapping the dimension and variables names of the source netCDF to the DISDRODB L0B standards.
   Variables not present the ``dict_names`` are dropped from the dataset.
   Variables specified in ``dict_names`` but missing in the dataset, are added as NaN arrays.
   Here is an example of dict_names:

   .. code-block:: python

       dict_names = {
           # Dimensions
           "timestep": "time",
           "diameter_bin": "diameter_bin_center",
           "velocity_bin": "velocity_bin_center",
           # Variables
           "reflectivity": "reflectivity_32bit",
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

We describe here the steps required to create a reader for your raw text files.
To share the reader with the community, please also read the `Contributing guide <contributors_guidelines.html>`_.


* `Step 1 <#step-1-add-the-raw-data-to-the-disdrodb-raw-archive>`_ : Set up the DISDRODB "Raw" directory structure
* `Step 2 <#step-2-define-the-reader-name>`_: Define the reader name
* `Step 3 <#step-3-define-the-stations-metadata-yaml-files>`_: Define and check the validity of the stations metadata
* `Step 4 <#step-4-analyse-the-data-and-define-the-reader-components>`_ : Analyse the raw data and implement the reader
* `Step 5 <#step-5-run-the-disdrodb-l0-processing>`_ : Run the DISDRODB L0 processing
* `Step 6 <#step-6-add-reader-testing-files>`_ : Create the reader test files



Step 1 : Add the raw data to the DISDRODB Raw archive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DISDRODB Raw archive has the following structure:

| 📁 DISDRODB
| ├── 📁 Raw
|    ├── 📁 <data_source>
|       ├── 📁 <campaign_name>
|           ├── 📁 data
|               ├── 📁 <station_name>
|                    ├── 📜 \*.\*  : raw files
|           ├── 📁 issue
|               ├── 📜 <station_name>.yml
|           ├── 📁 metadata
|               ├── 📜 <station_name>.yml


You can create the entire directory structure from scratch or you can clone the
`disdrodb-data <https://github.com/ltelab/disdrodb-data>`_ repository and
add the new required directories yourself.
The documentation on how to fork and clone a GitHub repository is available
in the `Contributors Guidelines <https://disdrodb.readthedocs.io/en/latest/contributors_guidelines.html#fork-the-repository>`_ section.

.. note::
	Guidelines for the naming of the ``<data_source>`` directory:

    * We use the institution name when campaign data spans more than 1 country.


    * We use country when all campaigns (or sensor networks) are inside a given country.


    * The ``<data_source>`` must be defined UPPER_CASE and without spaces.


.. note::
	Guidelines for the naming of the ``<campaign_name>`` directory:

    * The ``<campaign_name>`` must be defined UPPER_CASE and without spaces.


    * Avoid the usage of dash ( - ) and dots ( . ) to separate words. Use the underscore ( _ ) instead!

    * For short-term campaigns, we suggest adding the year of the campaign at the end (i.e. `EPFL_2009`)

.. note::
    Guidelines for the correct definition of the **metadata YAML files**:

    * For each ``<station_name>`` in the ``/data`` directory there must be an equally named ``<station_name>.yml`` file in the ``/metadata`` folder.

    * The metadata YAML file contains **relevant** information of the station (e.g. type of raw data, type of device, geolocation, ...) which is required for the correct processing and integration into the DISDRODB archive.

    * Read carefully `Step 2 <#step-2-define-the-metadata-of-the-stations>`_ to define the metadata correctly!

.. note::
    Guidelines for the definition of the **issue YAML files**:

    * The issue YAML files are optional (and if missing are initialized to be empty).

    * The issue YAML file of a station enable to specify the timesteps or time_periods to discard during the DISDRODB L0 processing of the raw data.

    * The issue YAML files of the entire DISDRODB archive are shared publicly and can be edited on the `disdrodb-data <https://github.com/ltelab/disdrodb-data>`_ repository.



Step 2 : Define the reader name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Since you aim to design a new reader, you can start by copy-pasting
`the reader_template.py <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/reader_template.py>`_
python file into the relevant ``disdrodb.l0.reader.<READER_DATA_SOURCE>`` directory.
If the ``<READER_DATA_SOURCE>`` for your reader does not yet exists, create a new directory.
Then rename the copied `reader_template.py` file with the name of your reader.
In order to guarantee consistency between readers, it is very important to follow a specific nomenclature.
Here in after we will refer to the reader name with ``<READER_NAME>``.

.. note::
    Guidelines for the definition of ``<READER_NAME>``:

    * The ``<READER_NAME>`` should corresponds to the name of the ``<CAMPAIGN_NAME>``.

    * The ``<READER_NAME>`` must be defined UPPER CASE, without spaces.

    * However, if a campaign requires different readers (because of different file formats or sensors), the ``<READER_NAME>`` is defined by adding a suffix preceded by an underscore indicating the stations or the sensor for which has been designed. Example: ``"RELAMPAGO_OTT"`` and ``"RELAMPAGO_RD80"``.


    * Have a look at the `pre-implemented DISDRODB readers <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers>`_ to grasp the terminology.


Despite being not yet implemented and working, your reader function can now be retrieved with:

.. code-block:: python

    from disdrodb.l0.L0_reader import get_station_reader

    disdrodb_dir = "<...>/DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    station_name = "STATION_NAME"
    reader = get_station_reader(
        disdrodb_dir=disdrodb_dir, data_source=data_source, campaign_name=campaign_name, station_name=station_name
    )


The reader will be customized successively, after having completed `Step 4 <#step-4-analyse-the-data-and-define-the-reader-components>`_


Step 3 : Define the stations metadata YAML files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The accurate definition of the stations metadata is essential for the correct
processing of the raw data.

Once you have placed all your raw data inside each <station_name> directory, you can copy-paste and
run the following code to generate a default metadata YAML file for each station:


.. code-block:: python

    from disdrodb.l0.metadata import create_campaign_default_metadata

    disdrodb_dir = "<...>/DISDRODB"
    campaign_name = "CAMPAIGN_NAME"
    data_source = "DATA_SOURCE"
    create_campaign_default_metadata(
        disdrodb_dir=disdrodb_dir,
        campaign_name=campaign_name,
        data_source=data_source,
    )


.. note::
    Run  ``create_campaign_default_metadata`` only if you are adding data for a new campaign.
    Otherwise copy-paste and modify existing metadata YAML files or alternatively
    use the ``disdrodb.l0.metadata.write_default_metadata`` to write a default metadata
    YAML file to a specific file path.

.. note::
    Do not eliminate metadata keys for which no information is available !


Now it's time to fill in the metadata information.
The list and description of the metadata is available in the
`Metadata <https://disdrodb.readthedocs.io/en/latest/metadata.html>`_  section.

There are 7 metadata keys for which is mandatory to specify the value :

* the ``data_source`` must be the same as the data_source where the metadata are located
* the ``campaign_name`` must be the same as the campaign_name where the metadata are located
* the ``station_name`` must be the same as the name of the metadata YAML file without the .yml extension
* the ``sensor_name`` must be one of the implemented sensor configurations. See ``disdrodb.available_sensor_name()``.
  If the sensor which produced your data is not within the available sensors, you first need to add the sensor
  configurations. For this task, read the section `Add new sensor configs <https://disdrodb.readthedocs.io/en/latest/sensor_configs.html>`_
* the ``raw_data_format`` must be either 'txt' or 'netcdf'. 'txt' if the source data are text/ASCII files. 'netcdf' if source data are netCDFs.
* the ``platform_type`` must be either 'fixed' or 'mobile'. If 'mobile', the DISDRODB L0 processing accepts latitude/longitude/altitude coordinates to vary with time.
* the ``reader`` name is essential to enable to select the correct reader when processing the station.


.. note::
    The **reader** key value must be defined with the pattern ``<READER_DATA_SOURCE>/<READER_NAME>``:

    - ``<READER_DATA_SOURCE>`` is the parent directory within the disdrodb software where the reader is defined. Typically it coincides with the ``<DATA_SOURCE>`` of the DISDRODB archive.

    - ``<READER_NAME>`` is the name of the python file where the reader is defined.


    For example, to use the GPM IFLOODS reader (defined at `disdrodb.l0.reader.GPM.IFLOODS.py <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers/GPM/IFLOODS.py>`_)
    to process the data, you specify the reader name ``GPM/IFLOODS``.

To check you are specifying the correct reader value in the metadata,
adapt the following piece of code to your use case:


.. code-block:: python

    from disdrodb.l0.L0_reader import get_reader_from_metadata_reader_key

    metadata_reader_value = "GPM/IFLOODS"
    reader = get_reader_from_metadata_reader_key(metadata_reader_value)
    print(reader)


Once you defined your metadata YAML files, check their validity by running:


.. code-block:: python

    from disdrodb.l0 import check_archive_metadata_compliance, check_archive_metadata_geolocation

    disdrodb_dir = "<...>/DISDRODB"
    check_archive_metadata_compliance(disdrodb_dir)
    check_archive_metadata_geolocation(disdrodb_dir)


Step 4 : Analyse the data and define the reader components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the DISDRODB directory structure, reader and metadata are set up, you can start analyzing the content of your data.
To facilitate the task, we provide you with the `reader_preparation.ipynb Jupyter Notebook <https://disdrodb.readthedocs.io/en/latest/reader_preparation.html>`_.
We highly suggest to copy the notebook and adapt it to your own data.

However, before starting adapting the Jupyter Notebook to your own data,
we highly suggest to first try it out with the sample lightweight dataset provided within the disdrodb repository.
Such step-by-step notebook tutorial is also visible in the `Tutorial: Reader preparation step-by-step <#tutorial-reader-preparation-step-by-step>`_ subsection
of this documentation.

These notebooks will guide you through the definition of the 4 relevant DISDRODB reader components:

* The ``glob_patterns`` to search for the data files within the ``.../data/<station_name>`` directory.

* The ``reader_kwargs`` dictionary guides the pandas dataframe creation.

For more information on the possible key-value arguments, read the `pandas <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_
documentation.

* The ``column_names`` list defines the column names of the read raw text file.

* The ``df_sanitizer_fun()`` function that defines the processing to apply on the read dataframe in order for the dataframe to match the DISDRODB standards.

The dataframe which is returned by the ``df_sanitizer_fun`` must have only columns compliant with the DISDRODB standards !

When this 4 components are correctly defined, they can be transcribed into the reader you defined in `Step 2 <#step-2-define-the-reader-name>`_.


Now you are ready to test the reader works properly and enable to process all stations data.


Step 5 : Run the DISDRODB L0 processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test the reader works properly, the easiest way now it's to run the
DISDRODB L0 processing of the stations for which you added the reader.

To run the processing of a single station, you can run:

    .. code-block:: bash

        run_disdrodb_l0_station <disdrodb_dir> <data_source> <campaign_name> <station_name> [parameters]


For example, to process the data of station 10 of the EPFL_2008 campaign, you would run:

    .. code-block:: bash

        run_disdrodb_l0_station /ltenas8/disdrodb-data/DISDRODB EPFL  EPFL_2008 10 --force True --verbose True --parallel False


If no problems arise, try to run the processing for all stations within a campaign, with:

	.. code-block:: bash

		run_disdrodb_l0 <disdrodb_dir> --data_sources <data_sources> --campaign_names <campaign_names> [parameters]

For example, to process all stations of the EPFL_2008 campaign, you would run:

	.. code-block:: bash

		run_disdrodb_l0 /ltenas8/disdrodb-data/DISDRODB --data_sources EPFL --campaign_names EPFL_2008 --force True --verbose True --parallel False


.. note::

    * For more details and options related to DISDRODB L0 processing, read the section `Run DISDRODB L0 Processing <https://disdrodb.readthedocs.io/en/latest/l0_processing.html>`_.


The DISDRODB L0 processing generates the DISDRODB `Processed` directories tree illustrated here below.

| 📁 DISDRODB
| ├── 📁 Processed
|    ├── 📁 <data_source>
|       ├── 📁 <campaign_name>
|           ├── 📁 L0A
|               ├── 📁 <station_name>
|                   ├── 📜 \*.parquet
|           ├── 📁 L0B
|               ├── 📁 <station_name>
|                    ├── 📜 \*.nc
|           ├── 📁 info
|           ├── 📁 logs
|               ├── 📁 L0A
|                   ├── 📁 <station_name>
|                        ├── 📜 \*.log
|                   ├── 📜 logs_problem_<station_name>.log
|                   ├── 📜 logs_summary_<station_name>.log
|               ├── 📁 L0B
|                   ├── 📁 <station_name>
|                        ├── 📜 \*.log
|                   ├── 📜 logs_problem_<station_name>.log
|                   ├── 📜 logs_summary_<station_name>.log
|           ├── 📁 metadata
|               ├── 📜 <station_name>.yml


If you inspect the ``logs/L0A`` and ``logs/L0B``, you will see the logging reports of the DISDRODB L0 processing.
For every raw file, a processing log is generated.
The ``logs_summary_<station_name>.log`` summarizes all the logs regarding the processing of a station.
Instead, if the ``logs_problem_<station_name>.log`` file is not present in the logs directory,
it means that the reader you implemented worked correctly, and no errors were raise by DISDRODB.

Otherwise, you need to investigate the reported errors, improve the readers and rerun the DISDRODB L0 processing.

Reiterate between Step 4 and Step 5 till the DISDRODB L0 processing does not raise errors :)


Step 6 : Add reader testing files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


If you arrived at this final step, it means that your reader is now almost ready to be shared with the community.

To ensure long-term maintainability of the DISDRODB project, we require to provide
a very small testing data sample composed of two raw files.
This enable our Continuous Integration (CI) testing routine to continuously check
that the reader you implemented will provide the expected results also
when someone else will add changes to the disdrodb codebase in the future.


.. note::
	The objective is to run every reader sequentially.
	Therefore, make sure to provide a very small test sample in order to limit the computing time.

	The size of the test sample must just be sufficient to guarantee the detection of errors due to code changes.
	A typical test file is composed of 2 stations, with two files and a couple of timesteps with measurements.


You should place you data and config files under the following directory tree:

| 📁 disdrodb/tests/pytest_files/check_readers/DISDRODB
| ├── 📁 Raw
|    ├── 📁 <DATA_SOURCE>
|       ├── 📁 <CAMPAIGN_NAME>
|           ├── 📁 issue
|               ├── 📜 <station_name>.yml
|           ├── 📁 metadata
|               ├── 📜 <station_name>.yml
|           ├── 📁 data
|               ├── 📁 <station_name>
|                   ├── 📜 <station_name>.\*
|           ├── 📁 ground_truth
|               ├── 📁 <station_name>
|                   ├── 📜 <station_name>.\*



The ``/data`` folder must contain your raw data files, while the ``/ground_truth`` folder must contain the corresponding ground truth files.

Once the reader is run with the raw data, the output files is compared to the ground truth files. If the files are identical, the reader is considered valid.



.. warning::

	Naming convention :

	``<data_source>``  : Name of the data source.

	* We use the institution name when campaign data spans more than 1 country.
	* We use country when all campaigns (or sensor networks) are inside a given country.
	* Must be in capital letter.
	* Must correspond to the name of the folder where the reader python file has been saved.
	* Example : `EPFL` or `ITALY` .


	``<campaign_name>``  : Name of the campaign.

	* Must be in capital letter.
	* Must correspond to the name of the reader python file.
	* Example : `LOCARNO2018` or `GID` .







Tutorial : Reader preparation step-by-step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please visit the following page to access a read-only tutorial notebook:


.. toctree::
   :maxdepth: 1

   reader_preparation

If you want to run an interactive notebook, you need to run jupyter notebook in your local machine. Proceed as follow :

1. Make sure you have the latest version of the code in your local folder.
See the git clone command in the `Installation for contributors <https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-contributors>`_ section.

2. Enter your project virtual environment or conda environment.
Please, refer to the `Installation for contributors <https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-contributors>`_ section if needed.

3. Navigate to the disdrodb folder.

4. Start the Jupyter Notebook with:

	.. code-block:: bash

		python -m notebook

	or

	.. code-block:: bash

		jupyter notebook

	This will open your default web browser with Jupyter Notebook on the main page.


5. Navigate to ``tutorials`` and double click on the ``reader_preparation.ipynb``.

6. Specify the IPython kernel on which to run the Jupyter Notebook.
To do so, first click on the top ``Kernel`` tab, then click on en ``Change Kernel``, and then select your environment.
If the environment is not available, close the Jupyter Notebook, type the following command and relaunch the notebook:

.. code-block:: bash

    python -m ipykernel install --user --name=<YOUR-ENVIRONMENT-NAME>


7. You can now start using the tutorial notebook.
