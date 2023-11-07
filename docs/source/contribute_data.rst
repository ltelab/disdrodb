==============================
How to Contribute New Data
==============================

Do you want to contribute your own data to to DISDRODB ? Great! You are in the right place !

The data contributor is asked to perform the following 4 tasks: 

- add the metadata of the stations he wish to contribute to the DISDRODB Metadata Archive,
- implement and add to the disdrodb python package the reader enabling the processing of the raw data to DISDRODB L0 products, 
- upload the raw data on a remote data repository (e.g., Zenodo, Figshare, etc.),
- test that the download and DISDRODB L0 processing of the stations he/she contributed is working correctly.

Before proceeding, you need to start thinking about the ``<DATA_SOURCE>`` and ``<CAMPAIGN_NAME>`` names of your stations.
The name you adopt for the ``<DATA_SOURCE>`` and ``<CAMPAIGN_NAME>`` will be used to define: 
-  the name of the directories where the raw data and the metadata of your stations will be stored in the DISDRODB Archive.
-  the name of the DISDRODB reader you will implement for your data.

 .. note::
	Guidelines for the naming of the ``<DATA_SOURCE>``:

    * We use the institution name when campaign data spans more than 1 country.

    * We use country when all campaigns (or sensor networks) are inside a given country.

    * The ``<DATA_SOURCE>`` must be defined UPPER_CASE and without spaces.


.. note::
	Guidelines for the naming of the ``<CAMPAIGN_NAME>``:

    * The ``<CAMPAIGN_NAME>`` must be defined UPPER_CASE and without spaces.

    * Avoid the usage of dash ( - ) and dots ( . ) to separate words. Use the underscore ( _ ) instead!

    * For short-term campaigns, we suggest adding the year of the campaign at the end (i.e. `EPFL_2009`)


Here below we provide a detailed description of the steps to follow to contribute your data to DISDRODB:

* `Step 1 <#step1>`_: Fork and download the DISDRODB Metadata Archive
* `Step 2 <#step2>`_: Define the DISDRODB base directory
* `Step 3 <#step3>`_: Add metadata the local DISDRODB Data Archive
* `Step 4 <#step4>`_: Add the raw data the local DISDRODB Data Archive
* `Step 5 <#step5>`_: Fork and install the disdrodb python package in editable mode
* `Step 6 <#step6>`_: Define the reader name and add a prototype reader to the disdrodb python package
* `Step 7 <#step7>`_: Implement the reader for your data
* `Step 8 <#step8>`_: Run the DISDRODB L0 processing
* `Step 9 <#step9>`_: Add reader testing files to the disdrodb python package
* `Step 10 <#step10>`_: Upload your raw data on Zenodo and link it to the DISDRODB Decentralized Data Archive
* `Step 11 <#step11>`_: Test the download and DISDRODB L0 processing of the stations you just contributed

Before going down the road, please also have a look at the `Contributors Guidelines <contributors_guidelines.html>`_.

.. _step1:

Step 1: Fork and download the DISDRODB Metadata Archive
--------------------------------------------------------------

1. Go to the `DISDRODB metadata repository <https://github.com/ltelab/disdrodb-data>`__, fork the repository on your GitHub account and then clone the forked repository:

   .. code:: bash

      git clone https://github.com/<your_username>/disdrodb-data.git

2. Go inside the ``disdrodb-data`` directory where you have cloned the repository:

3. Create a new branch:

   .. code:: bash

      git checkout -b "add-metadata-<data_source>-<campaign_name>"

   .. note:: 
      The ``<data_source>`` and ``<campaign_name>`` should correspond to the ``<DATA_SOURCE>`` and ``<CAMPAIGN_NAME>`` of the station you aim to contribute.

4. Set the remote upstream branch:
 
    .. code:: bash
      
      git push --set-upstream origin "add-metadata-<data_source>-<campaign_name>"

5. Every time you will now ``git add *`` and ``git commit -m <describe-your-change>`` your changes, you will be able to push them to your forked repository with:

   .. code:: bash

      git push

6. When you want to show your changes to the DISDRODB maintainers, you will need to open a Pull Request.
   To do so, go to the `GitHub disdrodb-data repository <https://github.com/ltelab/disdrodb-data>`__, open the Pull Request and ask for a review.

   For more information on GitHub Pull Requests, read the
   `"Create a pull request documentation" <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__.

   If you struggle with this process, do not hesitate to raise an `issue <https://github.com/ltelab/disdrodb-data/issues/new/choose>`__ 
   or ask in the `disdrodb slack channel <https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA>`__ so that we can help !

.. _step2:

Step 2: Define the DISDRODB base directory 
--------------------------------------------

Here we define the DISDRODB base directory of the archive. 
The directory path is saved into the DISDRODB configuration file,
which is used by the disdrodb python package to locate the DISDRODB archive.
On Windows, the DISDRODB base directory will have a path ending by `"\DISDRODB"`,  while on Mac/Linux, it will have a path ending by `"/DISDRODB"`

.. code:: python 

   import disdrodb
   base_dir = "<path_to>/disdrodb-data/DISDRODB"
   disdrodb.define_configs(zenodo_token="<your zenodo token>",
                           zenodo_sandbox_token="<your zenodo sandbox token>")


.. _step3:

Step 3: Add metadata the local DISDRODB Data Archive
-----------------------------------------------------------

Now let's create the directory structure and the defaults metadata files for the stations you wish to contribute.
If you contribute multiple stations, just rerun the following command for each station. 
   
.. code:: bash

   disdrodb_initialize_station_files <DATA_SOURCE> <CAMPAIGN_NAME> <STATION_NAME>


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


Go in the ``disdrodb-data/DISDRODB/Raw/<DATA_SOURCE>/<CAMPAIGN_NAME>/metadata/`` directory and start editing the metadata files 
of the stations you wish to contribute. 
The metadata YAML file contains **relevant** information of the station (e.g. type of raw data, type of device, geolocation, ...) which is 
required for the correct processing and integration into the DISDRODB archive.

The list and description of the metadata is available in the `Metadata <https://disdrodb.readthedocs.io/en/latest/metadata.html>`_ section.

There are 7 metadata keys for which is mandatory to specify the value:

* the ``data_source`` must be the same as the data_source where the metadata are located
* the ``campaign_name`` must be the same as the campaign_name where the metadata are located
* the ``station_name`` must be the same as the name of the metadata YAML file without the .yml extension
* the ``sensor_name`` must be one of the implemented sensor configurations. See ``disdrodb.available_sensor_names()``.
  If the sensor which produced your data is not within the available sensors, you first need to add the sensor
  configurations. For this task, read the section `Add new sensor configs <https://disdrodb.readthedocs.io/en/latest/sensor_configs.html>`_
* the ``raw_data_format`` must be either 'txt' or 'netcdf'. 'txt' if the source data are text/ASCII files. 'netcdf' if source data are netCDFs.
* the ``platform_type`` must be either 'fixed' or 'mobile'. If 'mobile', the DISDRODB L0 processing accepts latitude/longitude/altitude coordinates to vary with time.
* the ``reader`` name is essential to enable to select the correct reader when processing the station.

Please take care of the following points when editing the metadata files:

*  Do not eliminate metadata keys for which no information is available !
*  You will define the ``reader`` name in Step 6 along with the implementation of the reader
*  The station metadata YAML file must keep the name of the station (i.e. ``<station_name>.yml``)
*  For each ``<station_name>`` directory in the ``/data`` folder there must be an equally named ``<station_name>.yml`` file in the ``/metadata`` folder.

When you are done with the editing of the metadata files, please run the following command to check that the metadata files are valid:

   .. code:: bash

      disdrodb_check_metadata_archive --raise_error=False


The only error you should temporary get is the one related to the missing value of the ``reader`` key !

.. _step4:

Step 4: Add the raw data the local DISDRODB Data Archive
---------------------------------------------------------------

It's now time to move the raw data of each station into the corresponding ``disdrodb-data/DISDRODB/Raw/<DATA_SOURCE>/<CAMPAIGN_NAME>/data/<STATION_NAME>`` directory.

Once done, you are mostly ready for the next step: implementing the DISDRODB reader for your data.

.. _step5:

Step 5: Fork and install the disdrodb python package in editable mode 
-----------------------------------------------------------------------

But first, let's fork and install the `disdrodb python package  <https://github.com/ltelab/disdrodb>`__ in editable mode,
so that you will be able to share the reader you are going to develop with the community.

To do so, follow the steps below:

1. Go to the `disdrodb python package repository <https://github.com/ltelab/disdrodb>`__, fork the repository on your GitHub account and then clone the forked repository:

   .. code:: bash

      git clone https://github.com/<your_username>/disdrodb.git

2. Go inside the ``disdrodb`` directory where you have cloned the repository

3. Create a new branch where you will develop the reader for your data:

   .. code:: bash

      git checkout -b "reader-<data_source>-<campaign_name>"


4. Set the remote upstream branch:
 
    .. code:: bash
      
      git push --set-upstream origin "reader-<data_source>-<campaign_name>"

5. Every time you will now ``git add *`` and ``git commit -m <describe-your-change>`` your changes, you will be able to push them to your forked repository with:

   .. code:: bash

      git push


6. When you want to show your changes to the DISDRODB maintainers, you will need to open a Pull Request.
   To do so, go to the `GitHub disdrodb repository <https://github.com/ltelab/disdrodb>`__, open the Pull Request and ask for a review.

   For more information on GitHub Pull Requests, read the
   `"Create a pull request documentation" <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__.

   If you struggle with this process, do not hesitate to raise an `issue <https://github.com/ltelab/disdrodb/issues/new/choose>`__ 
   or ask in the `disdrodb slack channel <https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA>`__ so that we can help !


7. Finally, install the disdrodb python package in editable mode using:

   .. code:: bash

      pip install -e .


.. _step6:

Step 6: Define the reader name and add a prototype reader to the disdrodb python package
-------------------------------------------------------------------------------------------

DISDRODB readers are python functions that enable to read the raw data of a station. 
DISDRODB readers are located inside the disdrodb python package at `disdrodb.l0.reader.<READER_DATA_SOURCE>.<READER_NAME>.py <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers>`_

In order to guarantee consistency between DISDRODB readers, it is very important to follow a specific nomenclature for ``<READER_NAME>`` and ``<READER_DATA_SOURCE>``

The guidelines for the definition of ``<READER_NAME>`` are:

* The ``<READER_NAME>`` should corresponds to the name of the ``<CAMPAIGN_NAME>``.

* The ``<READER_NAME>`` must be defined UPPER CASE, without spaces.

* However, if a campaign requires different readers (because of different file formats or sensors), the ``<READER_NAME>`` is defined by adding a suffix preceded by an underscore indicating the stations or the sensor for which has been designed. Example: ``"RELAMPAGO_OTT"`` and ``"RELAMPAGO_RD80"``.

* Have a look at the `pre-implemented DISDRODB readers <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers>`_ to grasp the terminology.

The ``<READER_DATA_SOURCE>`` name typically coincides with the station ``<DATA_SOURCE>`` name.

Since you aim to design a new reader, you can start by copy-pasting
`the reader_template.py <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/reader_template.py>`_
python file into the relevant ``disdrodb.l0.reader.<READER_DATA_SOURCE>`` directory and rename it as ``<READER_NAME>.py``.

If the ``<READER_DATA_SOURCE>`` for your reader does not yet exists, create a new directory.

Once the reader template has been copied and renamed in the appropriate location of the disdrodb package, 
it's time to **update the value of the ``reader`` key in the metadata files** !!! 

The ``reader`` key value must be defined with the pattern ``<READER_DATA_SOURCE>/<READER_NAME>`` where:

* ``<READER_DATA_SOURCE>`` is the parent directory within the disdrodb software where the reader is defined. Typically it coincides with the ``<DATA_SOURCE>`` of the station.

* ``<READER_NAME>`` is the name of the python file where the reader is defined.

For example, to use the `disdrodb.l0.reader.GPM.IFLOODS.py reader <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers/GPM/IFLOODS.py>`_
to process the data, you specify the ``reader`` name ``GPM/IFLOODS``.

To check you are specifying the correct ``reader`` value in the metadata, adapt the following piece of code to your reader name and run it:
``get_reader_function_from_metadata_key`` should return the reader function:

.. code-block:: python

    from disdrodb.l0.l0_reader import get_reader_function_from_metadata_key

    reader_name = "GPM/IFLOODS" # <READER_DATA_SOURCE>/<READER_NAME>
    reader = get_reader_function_from_metadata_key(reader_name)
    print(reader)


If you updated the station metadata files, your reader function should also now be retrievable with the following function:

.. code-block:: python

    from disdrodb.l0.l0_reader import get_station_reader_function

    campaign_name = "<CAMPAIGN_NAME>"
    data_source = "<DATA_SOURCE>"
    station_name = "<STATION_NAME>"
    reader = get_station_reader_function(
        data_source=data_source,
        campaign_name=campaign_name, 
        station_name=station_name
    )

Once you updated your metadata YAML files, check once again the validity of the metadata by running:

   .. code:: bash

      disdrodb_check_metadata_archive


At this point, no error and printed message should appear !!!

If you have any question at this point, you are encountering some issues, or you just want to let the DISRODB maintainers know that you are working on the 
implementation of a reader for your data, just  ``git add *``, ``git commit -m <describe-your-change>``, ``git push`` your code changes ... 
and open a Pull Request in the `GitHub disdrodb repository <https://github.com/ltelab/disdrodb>`__ and `GitHub disdrodb-data repository <https://github.com/ltelab/disdrodb-data>`__ 
so that we keep track of your work and we can help you if needed !

.. _step7:

Step 7: Implement the reader for your data
---------------------------------------------

Once the DISDRODB directory structure, reader and metadata are set up, you can start analyzing the content of your data.
To facilitate the task, we provide you with the `reader_preparation Jupyter Notebook Tutorial <https://disdrodb.readthedocs.io/en/latest/reader_preparation.html>`_.
We highly suggest to copy the notebook and adapt it to your own data.

However, before starting adapting the Jupyter Notebook to your own data,
we highly suggest to first try it out with the sample lightweight dataset provided within the disdrodb repository.
Such step-by-step notebook tutorial is also visible in the `Step-by-step reader preparation <#step-by-step-reader-preparation>`_ subsection
of this documentation.

This notebook will guide you through the definition of the 4 relevant DISDRODB reader components:

* The ``glob_patterns`` to search for the data files within the ``.../data/<station_name>`` directory.

* The ``reader_kwargs`` dictionary guides the pandas dataframe creation.

For more information on the possible key-value arguments, read the `pandas <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_
documentation.

* The ``column_names`` list defines the column names of the read raw text file.

* The ``df_sanitizer_fun()`` function that defines the processing to apply on the read dataframe in order for the dataframe to match the DISDRODB standards.

The dataframe which is returned by the ``df_sanitizer_fun`` must have only columns compliant with the DISDRODB standards !

When this 4 components are correctly defined, they can be transcribed into the reader you defined in `Step 7 <#step7>`_

Now you are ready to test the reader works properly and enable to process all stations data.

If you arrived at this point and you didn't open yet a Pull Request in the `GitHub disdrodb repository <https://github.com/ltelab/disdrodb>`__, do it now so
that the DISDRODB maintainers can review your code and help you with the final steps !

.. _step8:

Step 8: Run the DISDRODB L0 processing
---------------------------------------

To test the reader works properly, the easiest way now it's to run the
DISDRODB L0 processing of the stations for which you added the reader.

To run the processing of a single station, you can run:

    .. code-block:: bash

        disdrodb_run_l0_station <data_source> <campaign_name> <station_name> [parameters]


For example, to process the data of station 10 of the EPFL_2008 campaign, you would run:

    .. code-block:: bash

        disdrodb_run_l0_station EPFL  EPFL_2008 10 --force True --verbose True --parallel False


If no problems arise, try to run the processing for all stations within a campaign, with:

	.. code-block:: bash

		disdrodb_run_l0 --data_sources <data_sources> --campaign_names <campaign_names> [parameters]

For example, to process all stations of the EPFL_2008 campaign, you would run:

	.. code-block:: bash

		disdrodb_run_l0 --data_sources EPFL --campaign_names EPFL_2008 --force True --verbose True --parallel False


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
Often, the errors arise from raw text files which are empty or corrupted. In such case, simply remove or sanitize the files.

Reiterate between `Step 4 <#step4>`_  and `Step 5 <#step5>`_ till the DISDRODB L0 processing does not raise errors :)

If you arrived at this point and you didn't open yet a Pull Request in the `GitHub disdrodb repository <https://github.com/ltelab/disdrodb>`__, do it now so
that the DISDRODB maintainers can review your code and help you with the final steps !

.. _step9:

Step 9: Add reader testing files to the disdrodb python package
-------------------------------------------------------------------

If you arrived at this final step, it means that your reader is now almost ready to be shared with the community.

To ensure long-term maintainability of the DISDRODB project, we kindly ask you to provide
a very small testing data sample composed of two raw files.
This enable our Continuous Integration (CI) testing routine to continuously check
that the reader you implemented will provide the expected results also
when someone else will add changes to the disdrodb codebase in the future.

.. note::
	The objective is to run every reader sequentially.
	Therefore, make sure to provide a very small test sample (a few KB in size) in order to limit the computing time.

	The size of the test samples must just be sufficient to guarantee the detection of errors due to code changes.
	The test samples are typically composed by two files and a couple of timesteps with measurements.

You should place you data and config files under the following directory tree:

| 📁 disdrodb/tests/data/check_readers/DISDRODB
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

If you arrived at this point and you didn't open yet a Pull Request in the `GitHub disdrodb repository <https://github.com/ltelab/disdrodb>`__ 
and in the `Github DISDRODB Metadata Repository <https://github.com/ltelab/disdrodb-data>`__, do it now so
that the DISDRODB maintainers can review your code and help you with the final steps !

.. note:: 
   To open a Pull Request in the `Github DISDRODB Metadata Repository <https://github.com/ltelab/disdrodb-data>`__, you need to  ``git push`` the changes 
   of your local ``disdrodb-data`` directory. 

.. note:: 
   To open a Pull Request in the `GitHub disdrodb repository <https://github.com/ltelab/disdrodb>`__, you need to ``git push`` the changes 
   of your local ``disdrodb`` python package directory. 

.. _step10:

Step 10: Upload your raw data on Zenodo and link it to the DISDRODB Decentralized Data Archive
-------------------------------------------------------------------------------------------------

We provide users with a code to easily upload their stations raw data to `Zenodo <https://zenodo.org/>`_.

If you aim to upload the data of a single station, run: 

.. code:: bash

   disdrodb_upload_station <DATA SOURCE> <CAMPAIGN_NAME> <STATION_NAME> --platform zenodo.sandbox --force False


If ``--platform zenodo.sandbox`` is specified, you are actually uploading the data in the Zenodo sandbox testing environment.
It's good practice to first upload the station there, to check that everything works fine (see `Step 11 <#step11>`_ below),
and then upload the data in the production environment using ``--platform zenodo``

In order to upload the data to Zenodo, you need to specify the Zenodo tokens into the DISDRODB configuration file with: 

.. code:: python 

   import disdrodb

   disdrodb.define_configs(zenodo_token="<your zenodo token>",
                           zenodo_sandbox_token="<your zenodo sandbox token>")


To generate the tokens, for `Zenodo go here <https://zenodo.org/account/settings/applications/tokens/new/>`_, while for 
`Zenodo Sandbox go here <https://sandbox.zenodo.org/account/settings/applications/tokens/new/>`_. When generating the tokens,
you can choose the name you want (i.e. DISDRODB), but you need to select the ``deposit:actions`` and ``deposit:write`` scopes.

When the token is generated, you will see something similar to the following:
.. image:: /static/zenodo.png


When the command  ``disdrodb_upload_station`` is executed, the data are automatically uploaded on Zenodo. 
A link will be displayed that the user must use to go to the Zenodo web interface to manually publish the data. 
Please select the community ``DISDRODB`` (see top blue button) before publishing the data !

.. image:: /static/zenodo_publishing_data.png

If you are uploading multiple stations, you can an overview of the data still waiting for publication at: 

* `https://sandbox.zenodo.org/me/uploads for the Zenodo Sandbox repository <https://sandbox.zenodo.org/me/uploads>`_
  
* `https://zenodo.org/me/uploads for the Zenodo repository <https://zenodo.org/me/uploads>`_

Note that: 

* when the data are uploaded on Zenodo, the metadata key ``disdrodb_data_url`` of the station is automatically 
  updated with the Zenodo URL where the station data are stored (and can be downloaded)

* if the ``authors``, ``authors_url`` and ``institution`` DISDRODB metadata keys are correctly specified 
  (i.e. each author information is comma-separated), these keys values are automatically added to the Zenodo metadata 
  required for the publication of the data.

* if the station data is not yet published on Zenodo, the data can still already be downloaded (i.e. for testing purposes).


If you feel safe about your data and the whole procedure, you can also use the command below to upload all stations of a given campaign. 

.. code:: bash

   disdrodb_upload_archive --data_sources <DATA SOURCE> --campaign_name> <CAMPAIGN_NAME> --platform zenodo.sandbox --force False 

Consider that if you previously uploaded data on Zenodo Sandbox for testing purposes, you need to specify ``--force True`` 
when uploading data to the official Zenodo repository ! 

.. _step11:

Step 11: Test the download and DISDRODB L0 processing of the stations you just contributed
-------------------------------------------------------------------------------------------

To test that the data upload has been successfull, you can try to download the data and run the DISDRODB L0 processing. 
However you **must NOT perform this test in the disdrodb-data directory you where working till now** because you would risk to 
overwrite/delete the data you just uploaded on Zenodo.

We provide this python script that should enable you to test safely the whole procedure. 

.. code:: python 

   import disdrodb 
   from disdrodb.l0.io import prepare_test_archive
   
   data_source="<your_data_source>"
   campaign_name="<your_campaign>"
   station_name="<your_campaign>"

   test_base_directory = "/tmp/DISDRODB" 
   prepare_test_archive(test_base_directory=test_base_directory, 
                        data_source=data_source, 
                        campaign_name=campaign_name,
                        station_name=station_name,
   )

   disdrodb.download_station(base_dir=test_base_directory,
                             data_source=data_source, 
                             campaign_name=campaign_name,
                             station_name=station_name)
   disdrodb.run_l0_station(base_dir=test_base_directory,
                           data_source=data_source, 
                           campaign_name=campaign_name,
                           station_name=station_name,
                           debugging_mode=True, # to speed up
   ) 

When the script finish, check that the content in the ``test_base_directory`` directory is what you expected to be. 

If everything looks as expected ... congratulations, you made it !!! 

Your Pull Requests will likely be immediately merged and the data and reader will be available to the DISDRODB community.

