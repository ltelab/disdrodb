=========================
Quick Start
=========================

In this section, we show how to quickly download raw disdrometer data
from the DISDRODB Decentralized Data Archive and generate DISDRODB products on your local machine.

However, before you begin, you need to download the DISDRODB Metadata Archive on your local machine.
This archive lists the available stations and contains pointers to the online repositories
that store each station's raw data.

Also, to follow this tutorial, activate a virtual environment with the disdrodb package installed.
Refer to the `Installation <https://disdrodb.readthedocs.io/en/latest/installation.html>`_ section
for details on setting up and activating the environment.


1. Download the DISDRODB Metadata Archive
-----------------------------------------------

Navigate to the directory where you want to store the DISDRODB Metadata Archive:

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

The disdrodb software requires two directory paths:
the local DISDRODB Metadata Archive and the local DISDRODB Data Archive,
where raw station data and generated products will be stored.

The disdrodb software will look for a configuration file called ``.config_disdrodb.yml``
in your home directory (i.e. ``~/.config_disdrodb.yml``).

Within the disdrodb package, we refer to the base directory of
the local DISDRODB Metadata Archive with the argument ``metadata_archive_dir``, while
to the base directory of the local DISDRODB Data Archive with the argument ``data_archive_dir``.

The ``metadata_archive_dir`` path corresponds to the ``DISDRODB`` directory within the ``DISDRODB-METADATA`` archive.
The ``data_archive_dir`` path corresponds to ``DISDRODB`` directory of choice where
all DISDRODB products will be saved.



To facilitate the creation of the DISDRODB Configuration File, you can adapt and
run in python the following code snippet.
Note that on Windows paths must end with ``\DISDRODB``,
while on macOS/Linux they must end with ``/DISDRODB``.

.. code:: python

    import disdrodb

    metadata_archive_dir = "<path_to>/DISDRODB-METADATA/DISDRODB"
    data_archive_dir = "<path_of_choice_to_the_local_data_archive>/DISDRODB"
    disdrodb.define_configs(metadata_archive_dir=metadata_archive_dir, data_archive_dir=data_archive_dir)

Running this code snippet writes a ``.config_disdrodb.yml`` file to your home directory
(for example ``~/.config_disdrodb.yml``), which disdrodb uses as the default configuration.

After closing and reopening your Python session,
running the following code should display the paths you defined in the configuration file:

.. code:: python

    import disdrodb

    print("DISDRODB Metadata Archive Directory: ", disdrodb.get_metadata_archive_dir())
    print("DISDRODB Data Archive Directory: ", disdrodb.get_data_archive_dir())


You can also verify and print the default DISDRODB Metadata Archive and Data Archive directories by typing the following command in the terminal:

.. code:: bash

   disdrodb_data_archive_directory
   disdrodb_metadata_archive_directory


Although not recommended for beginner users, you also have the option to define the DISDRODB Data and Metadata Archive directories using environment variables.
This can be done by setting the ``DISDRODB_DATA_ARCHIVE_DIR`` and ``DISDRODB_METADATA_ARCHIVE_DIR`` variables either directly in your terminal or by adding them to your
``.bashrc`` (or equivalent shell configuration) script.
To set them in the terminal, you can use the following commands:

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


By periodically updating the DISDRODB Metadata Archive,
you can download new stations as they become available.

To download all raw data in the DISDRODB Decentralized Data Archive, run:

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

The DISDRODB L0 processing chain converts raw data into a standardized format,
saving each day's data in a NetCDF file.

The DISDRODB L1 processing chain ingests the L0C product files,
performing quality checks, data homogenization, and filtering.

To know more about the various DISDRODB products, please refer to the `DISDRODB Products <https://disdrodb.readthedocs.io/en/latest/products.html>`_ section.

Generating these products requires only two simple commands:

.. code:: bash

   disdrodb_run_l0_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True
   disdrodb_run_l1_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True

For illustrative purposes, this example processes 3 raw files (``--debugging_mode True``)
with disabled parallelism (``--parallel False``) and verbose output  (``--verbose True``).

Please note that parallel (multi)processing is enabled by default (``--parallel True``).
If you want to keep track of the processing, the ``logs`` directory in the DISDRODB Data Archive
allows you to check the processing status of each file.

You can open the ``logs`` directory typing the following command in the terminal:

.. code:: bash

    disdrodb_open_logs_directory EPFL HYMEX_LTE_SOP3 10


5. Open and analyze the DISDRODB product files
----------------------------------------------

The disdrodb ``open_dataset`` function lets you lazily open all station files for
a DISDRODB product as an ``xarray.Dataset`` (or ``pandas.DataFrame`` for the L0A product).

.. code:: python

    import disdrodb

    # Define station arguments
    data_source = "EPFL"
    campaign_name = "HYMEX_LTE_SOP3"
    station_name = "10"

    # Open all station files of a given DISDRODB product
    ds = disdrodb.open_dataset(
        product="L0C",
        # Station arguments
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    ds

Alternatively, ``find_files`` lists all station files for a given product,
letting you open them as you wish.

.. code:: python

    import disdrodb
    import xarray as xr

    # Define station arguments
    data_source = "EPFL"
    campaign_name = "HYMEX_LTE_SOP3"
    station_name = "10"

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


We hope this tutorial helps you get started quickly with the disdrodb software.

To contribute new stations to the DISDRODB Decentralized Data Archive,
see the `How to contribute new data <https://disdrodb.readthedocs.io/en/latest/contribute_data.html>`_ guidelines.

To learn more about DISDRODB products, see the `Products <https://disdrodb.readthedocs.io/en/latest/products.html>`_ section.
To customize the processing chain, see the `DISDRODB Archive Processing <https://disdrodb.readthedocs.io/en/latest/processing.html>`_ section.
