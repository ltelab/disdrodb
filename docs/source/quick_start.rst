=========================
Quick Start
=========================

In this section, we describe how to quickly start to download disdrometer raw data 
from the DISDRODB Decentralized Data Archive and generate DISDRODB products on your local machine.

First, is however necessary to download on your local machine the DISDRODB Metadata Archive, 
which list the available stations and contains the pointers to the online data repositiores where
the raw data of the DISDRODB stations are stored.

Additionnaly, to follow this tutorial, you should be in a virtual environment with the disdrodb package installed!
Refers to the `Installation <https://disdrodb.readthedocs.io/en/latest/installation.html>`_ section for more details
on how to set-up and activate the virtual environment.


1. Download the DISDRODB Metadata Archive
-----------------------------------------------

First travel to the directory where you want to store the DISDRODB Metadata Archive with:

.. code:: bash

   cd  /path/to/directory/where/to/store/the/metadata/archive


Then clone the DISDRODB Metadata Archive repository with:

.. code:: bash

   git clone https://github.com/ltelab/DISDRODB-METADATA.git

This will create a directory called ``DISDRODB-METADATA``.

.. note:: The DISDRODB Metadata Archive is often updated with new stations or metadata.
          Therefore, we recommend to regularly update your local DISDRODB Metadata Archive by 
          running :code:`git pull` inside the ``DISDRODB-METADATA`` directory.


2. Define the DISDRODB Configuration File
------------------------------------------

The disdrodb software needs to know where the local DISDRODB Metadata Archive
is stored on your local machine, as well as where you want to download the raw stations data
as well as where to save the DISDRODB products you will generate. 

The disdrodb software will look for a configuration file called ``.config_disdrodb.yml`` 
in your home directory (i.e. ``~/.config_disdrodb.yml``). 

Within the disdrodb package, we refer to the base directory of
the local DISDRODB Metadata Archive with the argument ``metadata_archive_dir``, while 
to the base directory of the local DISDRODB Data Archive with the argument ``data_archive_dir``. 

The ``metadata_archive_dir`` path corresponds to the ``DISDRODB`` directory within the ``DISDRODB-METADATA`` archive. 
The ``data_archive_dir`` path corresponds to ``DISDRODB`` directory of choice where 
all DISDRODB products will be saved. 



To facilitate the creation of the DISDRODB Configuration File, you can adapt and run in python the following code snippet.
Please note that on Windows, these paths must end with ``"\DISDRODB"``,  while on Mac/Linux they must end with ``"/DISDRODB"``.

.. code:: python

    import disdrodb

    metadata_archive_dir  = "<path_to>/DISDRODB-METADATA/DISDRODB"
    data_archive_dir = "<path_of_choice_to_the_local_data_archive>/DISDRODB"
    disdrodb.define_configs(metadata_archive_dir=metadata_archive_dir,
                           data_archive_dir=data_archive_dir)

By running this command, the disdrodb software will write a ``.config_disdrodb.yml`` file into your home directory (i.e. ``~/.config_disdrodb.yml``)
that will be used as default configuration file when running the disdrodb software.

If you **now close your python session and reopen a new one**, if you will run the following code snippet, you 
should get the ``metadata_archive_dir`` and ``data_archive_dir`` paths you just defined in the DISDRODB Configuration File: 

.. code:: python

    import disdrodb

    print("DISDRODB Metadata Archive Directory: ", disdrodb.get_metadata_archive_dir()) 
    print("DISDRODB Data Archive Directory: ", disdrodb.get_data_archive_dir()) 


Alternatively, you can also define the DISDRODB Data and Metadata Archive directories as environment variables by 
specifying the ``DISDRODB_DATA_ARCHIVE_DIR`` and ``DISDRODB_METADATA_ARCHIVE_DIR`` variables in your terminal or ``.bashrc`` script.
In the terminal, you can type the following command:

.. code:: bash

   export DISDRODB_DATA_ARCHIVE_DIR="<path_of_choice_to_the_local_data_archive>/DISDRODB"
   export DISDRODB_METADATA_ARCHIVE_DIR="<path_to>/DISDRODB-METADATA/DISDRODB"

.. note:: It is important to remember that the environment variables ``DISDRODB_DATA_ARCHIVE_DIR`` and ``DISDRODB_METADATA_ARCHIVE_DIR``, if defined, 
   will take priority over the default path defined in the ``.config_disdrodb.yml`` file.


3. Download the DISDRODB Raw Data Archive
-------------------------------------------

The DISDRODB Metadata Archive holds the required stations information to download raw data from the DISDRODB Decentralized Data Archive.

Currently, only a subset of stations is available in the DISDRODB Decentralized Data Archive, but the community is working to make all the stations available.

You can check the stations currently available for download by running the following command:

.. code:: python

   import disdrodb

   disdrodb.available_stations(available_data=True)


By updating from time-to-time the DISDRODB Metadata Archive, you will be able to download new stations as they become available.

To download all raw data stored into the DISDRODB Decentralized Data Archive, you just have to run the following command:

.. code:: bash

   disdrodb_download_archive  --data_sources <data_source> --campaign_names <campaign_name> --station_names <station_name> --force false

The ``data_sources``, ``campaign_names`` and ``station_names`` parameters are optional and are meant to restrict the download to a specific set of
data sources, campaigns, and/or stations.

Parameters:

-  ``data_sources`` (optional): Station data sources.
-  ``campaign_names`` (optional): Station campaign names.
-  ``station_names`` (optional): Name of the stations.
-  ``force`` (optional, default = ``False``): a boolean value indicating whether existing files should be overwritten.

To download data from multiple data sources, campaigns, or stations, please provide a space-separated string of
the data sources, campaigns or stations you require.

For example:

* if you want to download all EPFL and NASA data use ``--data_sources "EPFL NASA"``,

* if you want to download stations of specific campaigns, use ``--campaign_names "HYMEX_LTE_SOP3 HYMEX_LTE_SOP4"``.

* if you want to download stations named in a specific way, use ``--station_names "station_name1 station_name2"``.

As an example for this tutorial, we will just download the data of a single station by running the following command in the terminal: 

.. code:: bash

   disdrodb_download_station EPFL HYMEX_LTE_SOP3 10 

Please note that ``EPFL HYMEX_LTE_SOP3 10`` arguments refers to the ``data_source``, ``campaign name`` and ``station name`` respectively. 


4. Generate DISDRODB L0 and L1 products
----------------------------------------------

Once the data are downloaded, we can start the generation of the DISDRODB L0 and L1 products.

The DISDRODB L0 processing chain convert the raw data into a standardized format, saving the raw data into a NetCDF file per day.

The DISDRODB L1 processing chain ingest the DISDRODB L0C product files and perform quality checks, data homogenization 
and data filtering.

To know more about the various DISDRODB products, please refer to the `DISDRODB Products <https://disdrodb.readthedocs.io/en/latest/products.html>`_ section.

The procedure to generate such products is very simple and just require typing the following two commands: 

.. code:: bash

   disdrodb_run_l0_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True
   disdrodb_run_l1_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True

For illustratory purposes, here we just process 3 raw files (``--debugging_mode True``). 
We also apply ``verbose`` processing, which requires disabling parallelism (``--parallel False``).

Please note that parallel (multi)processing is enabled by default (``--parallel True``).
If you want to keep track of the processing, the ``logs`` directory in the DISDRODB Data Archive
allows you to check the processing status of each file.

You can open the ``logs`` directory using the following command in python:

.. code:: python

   import disdrodb
   disdrodb.open_logs_directory(data_source="EPFL", campaign_name="HYMEX_LTE_SOP3", station_name="10")


5. Open and analyze the DISDRODB product files
----------------------------------------------

The disdrodb software ``open_dataset`` function enable to lazy open all station files of
a DISDRODB product into a ``xarray.Dataset`` (or ``pandas.DataFrame`` for the DISDRODB L0A product).

.. code:: python

   import disdrodb 

   # Define station arguments
   data_source="EPFL"
   campaign_name="HYMEX_LTE_SOP3"
   station_name="10"

   # Open all station files of a given DISDRODB product
   ds = disdrodb.open_dataset( 
      product="L0C",
      # Station arguments
      data_source=data_source,
      campaign_name=campaign_name,
      station_name=station_name,
   )
   ds

Alternatively, the disdrodb software ``find_files`` function allows to easily list all station files of a 
given DISDRODB product and then open the data as the user wish.

.. code:: python

   import disdrodb 
   import xarray as xr

   # Define station arguments
   data_source="EPFL"
   campaign_name="HYMEX_LTE_SOP3"
   station_name="10"

   # List all files 
   filepaths = disdrodb.find_files(
      product="L0C",
      data_source=data_source,
      campaign_name=campaign_name,
      station_name=station_name,
   )
   # Open a single file
   ds = xr.open_dataset(filepaths[0])
   ds


With this tutorial we hope you will be able to quickly start using the disdrodb software. 

If you wish to contribute new stations to the DISDRODB Decentralized Data Archive, please 
read the `how to contribute new data <https://disdrodb.readthedocs.io/en/latest/contribute_data.html>`_" guideline.

To know more about the various DISDRODB products, please refer to the
DISDRODB `Products <https://disdrodb.readthedocs.io/en/latest/products.html>`_ section, 
while to learn on how to customize the product processing chain, 
please refer to the `DISDRODB Archive Processing <https://disdrodb.readthedocs.io/en/latest/processing.html>`_ section. 
