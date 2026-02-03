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


4. Generate DISDRODB L0, L1 and L2E products
----------------------------------------------

Once the data are downloaded, we can start the generation of the DISDRODB L0, L1 and L2E products.

The DISDRODB L0 processing chain converts raw data into a standardized format,
saving each day's data in a NetCDF file.

The DISDRODB L1 processing chain ingests the L0C product files, aggregates data at user-defined temporal resolutions,
performs quality checks, data homogenization, and apply an hydrometeor classification algorithm that allows to
differentiate between rain, snow, mixed, and non-hydrometeor particles detected by the sensors.

The DISDRODB L1 product forms the basis for generating DISDRODB L2 products, which provide advanced retrievals of drop size distribution moments
and other microphysical parameters at user-defined temporal resolutions.

To know more about the various DISDRODB products and how to customize the processing,
please refer to the `DISDRODB Products <https://disdrodb.readthedocs.io/en/latest/products.html>`_ section.

Generating DISDRODB L0 and L1 products requires only two simple commands:

.. code:: bash

   disdrodb_run_l0_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True
   disdrodb_run_l1_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True

For illustrative purposes, this code snippet here above processes just 3 raw files (``--debugging_mode True``)
with disabled parallelism (``--parallel False``) and verbose output  (``--verbose True``).

To process all data for a given station use:

.. code:: bash

   disdrodb_run_l0_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True
   disdrodb_run_l1_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True
   disdrodb_run_l2e_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True

Alternatively, to create DISDRODB L0, L1 or L2E products for a specific station with a single command, you can also use:

.. code:: bash

   disdrodb_run_station EPFL HYMEX_LTE_SOP3 10 -p True -f True

``--force True`` forces the re-processing of products if they already exist on disk.

Please note that parallel (multi)processing is enabled by default (``--parallel True``).
If you want to keep track of the processing, you can open the Dask Dashboard at
`http://localhost:8787/status <http://localhost:8787/status>`_.

Alternatively, the ``logs`` directory in the DISDRODB Data Archive
allows you to check the processing status of each file.

You can open the ``logs`` directory typing the following command in the terminal:

.. code:: bash

    disdrodb_open_logs_directory EPFL HYMEX_LTE_SOP3 10


5. Open and analyze the DISDRODB L0 product files
---------------------------------------------------

The disdrodb ``open_dataset`` function lets you lazily open all station files for
a DISDRODB product as an ``xarray.Dataset`` (or ``pandas.DataFrame`` for the L0A product).
Here below we show how to open all files of the DISDRODB L0 product for a given station:

.. code:: python

    import disdrodb

    # Define station arguments
    data_source = "EPFL"
    campaign_name = "HYMEX_LTE_SOP3"
    station_name = "10"
    product = "L0C"

    # Open all station files of a given DISDRODB product
    ds = disdrodb.open_dataset(
        product=product,
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
    product = "L0C"

    # List all files
    filepaths = disdrodb.find_files(
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Open a single file
    ds = xr.open_dataset(filepaths[0])
    ds


.. warning::

    Users are expected to properly acknowledge the data they use by citing
    and referencing each station. The corresponding references, recommended
    citations, and DOIs are available in the DISDRODB netCDFs/xarray.Dataset
    global attributes, as well as in the DISDRODB Metadata Archive.


6. Open and analyze the DISDRODB L1 product files
---------------------------------------------------

Opening DISDRODB L1 product files is similar to opening L0 product files, but requires specifying the desired temporal resolution.

.. code:: python

    import disdrodb

    # Define station arguments
    data_source = "EPFL"
    campaign_name = "HYMEX_LTE_SOP3"
    station_name = "10"
    product = "L1"
    temporal_resolution = "1MIN"

    # Open all station files of the DISDRODB L1 product
    ds = disdrodb.open_dataset(
        product=product,
        # Station arguments
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product options
        temporal_resolution=temporal_resolution,
    )
    ds = ds.compute()

The DISDRODB L1 product includes hydrometeor classification and precipitation type variables.
These variables help differentiate between rain, snow, mixed, and non-hydrometeor particles detected by the sensors.

Here below we show how to select a specific hydrometeor type and plot the corresponding raw spectrum(s):

.. code:: python

    ds["precipitation_type"]
    print(ds["precipitation_type"].attrs)

    ds["hydrometeor_type"]
    print(ds["hydrometeor_type"].attrs)

    # Select timesteps with rain
    ds_rain = ds.isel(time=(ds["precipitation_type"] == 0))

    # Sum over time and plot the spectrum
    ds_rain.disdrodb.plot_spectrum()

    # Plot raw spectrum of the timestep with more drops
    ds_rain.isel(time=ds_rain["n_particles"].argmax().item()).disdrodb.plot_spectrum()

    # Select timesteps with likely graupel
    ds_graupel = ds.isel(time=(ds["hydrometeor_type"] == 8))
    ds_graupel.disdrodb.plot_spectrum()

    # Select timesteps with large hail
    ds_hail = ds.isel(time=(ds["flag_hail"] == 2))
    ds_hail.disdrodb.plot_spectrum()


6. Open and analyze the DISDRODB L2E product files
---------------------------------------------------

Opening DISDRODB L2E product files is similar to opening DISDRODB L1 product files:

.. code:: python

    import disdrodb

    # Define station arguments
    data_source = "EPFL"
    campaign_name = "HYMEX_LTE_SOP3"
    station_name = "10"
    product = "L2E"
    temporal_resolution = "1MIN"

    # Open all station files of the DISDRODB L2E product
    ds = disdrodb.open_dataset(
        product=product,
        # Station arguments
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product options
        temporal_resolution=temporal_resolution,
    )

The DISDRODB L2E product focuses on rainy timesteps and provides drop size distribution moments,
microphysical parameters and simulated polarimetric radar variables at various frequencies.


6. Explore available stations
------------------------------------

You can list all stations registered in the DISDRODB Archive using the ``disdrodb.available_stations`` function.

By default, the function returns all known stations, regardless of whether their raw data are currently available for download.
Setting ``available_data=True`` restricts the output to stations whose raw data are already available in the DISDRODB Decentralized Data Archive.
Note that some contributors have not yet made their data publicly available.

.. code:: python

    import disdrodb

    disdrodb.available_stations()  # available_data=False by default
    disdrodb.available_stations(available_data=True)


You can further filter the list of stations by sensor type and/or by the native measurement interval using the ``sensor_name`` and ``measurement_interval`` arguments:

.. code:: python

    import disdrodb

    disdrodb.available_stations(sensor_name="PWS100", available_data=True)
    disdrodb.available_stations(sensor_name="LPM", available_data=True)
    disdrodb.available_stations(sensor_name="PARSIVEL", measurement_interval=10, available_data=True)
    disdrodb.available_stations(sensor_name=["PARSIVEL", "PARSIVEL2"], measurement_interval=60, available_data=True)
    disdrodb.available_stations(sensor_name=["RD80"], measurement_interval=10, available_data=True)

Stations can also be filtered by data source, campaign name, or station name.
Multiple filters can be combined in a single call.


.. code:: python

    import disdrodb

    disdrodb.available_stations(data_sources=["ITALY", "EPFL"])
    disdrodb.available_stations(campaign_names="RELAMPAGO")
    disdrodb.available_stations(station_names=["TC-TO", "TC-AQ"], measurement_interval=60)


After downloading data and generating DISDRODB L1 or L2 products, you can list the stations you have processed by specifying the product type and temporal resolution.

.. code:: python

    import disdrodb

    disdrodb.available_stations(product="L1", temporal_resolution="1MIN")
    disdrodb.available_stations(sensor_name="PARSIVEL2", product="L1", temporal_resolution="1MIN")
    disdrodb.available_stations(sensor_name="PARSIVEL2", product="L2E", temporal_resolution="1MIN")


Please note that the ``temporal_resolution`` argument is mandatory when listing DISDRODB L1 and L2 products.


7 . What's next?
---------------------

We hope this tutorial helps you get started quickly with the disdrodb software.

To contribute new stations to the DISDRODB Decentralized Data Archive,
see the `How to contribute new data <https://disdrodb.readthedocs.io/en/latest/contribute_data.html>`_ guidelines.

To learn more about DISDRODB products, see the `Products <https://disdrodb.readthedocs.io/en/latest/products.html>`_ section.
To customize the processing chain, see the `DISDRODB Archive Processing <https://disdrodb.readthedocs.io/en/latest/processing.html>`_ section.
