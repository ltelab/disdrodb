.. _quick_start:

=========================
Quick Start
=========================

This quick start guide demonstrates how to download raw disdrometer data from the DISDRODB
Decentralized Data Archive and generate DISDRODB products on your local machine.

**Prerequisites**

- Virtual environment with DISDRODB installed (see :ref:`Installation <installation>`)
- DISDRODB Metadata Archive cloned locally
- Sufficient disk space for raw data and products

**What You'll Learn**

- Configure DISDRODB directories
- Download station data
- Generate DISDRODB L0, L1, and L2E products
- Open and analyze products with xarray
- Explore available stations


1. Download the DISDRODB Metadata Archive
-----------------------------------------------

The DISDRODB Metadata Archive contains station information, data sources, and pointers
to raw data in the Decentralized Data Archive.

Navigate to the directory where you want to store the DISDRODB Metadata Archive:

.. code:: bash

   cd /path/to/directory/where/to/store/the/metadata/archive

Clone the DISDRODB Metadata Archive repository:

.. code:: bash

   git clone https://github.com/ltelab/DISDRODB-METADATA.git

This creates a ``DISDRODB-METADATA`` directory.

.. note::

   **Git Required**: Ensure ``git`` is installed on your system.

   - Ubuntu/Debian: ``sudo apt-get install git``
   - macOS: ``brew install git`` or install Xcode Command Line Tools
   - Windows: Download from https://git-scm.com/

.. note:: The DISDRODB Metadata Archive is often updated with new stations or metadata.
          Therefore, we recommend to regularly update your local DISDRODB Metadata Archive by
          running :code:`git pull` inside the ``DISDRODB-METADATA`` directory.


2. Define the DISDRODB Configuration File
------------------------------------------

DISDRODB requires two directory paths:

- **Metadata Archive Directory** (``metadata_archive_dir``): Path to the ``DISDRODB`` subdirectory
  within your cloned ``DISDRODB-METADATA`` repository
- **Data Archive Directory** (``data_archive_dir``): Path where raw data and processed products will be stored

DISDRODB searches for a configuration file ``~/.config_disdrodb.yml`` in your home directory.

**Create Configuration File**

To facilitate the creation of the DISDRODB Configuration File, you can adapt and
run in python the following code snippet. See :func:`disdrodb.define_configs` for more details.
Note that on Windows paths must end with ``\DISDRODB``,
while on macOS/Linux they must end with ``/DISDRODB``.

.. code:: python

    import disdrodb

    metadata_archive_dir = "<path_to>/DISDRODB-METADATA/DISDRODB"
    data_archive_dir = "<path_of_choice_to_the_local_data_archive>/DISDRODB"
    disdrodb.define_configs(metadata_archive_dir=metadata_archive_dir, data_archive_dir=data_archive_dir)

This creates ``~/.config_disdrodb.yml`` in your home directory, which DISDRODB uses as the default configuration.

**Verify Configuration**

Restart your Python session and verify the configuration:

.. code:: python

    import disdrodb

    print("DISDRODB Metadata Archive Directory: ", disdrodb.get_metadata_archive_dir())
    print("DISDRODB Data Archive Directory: ", disdrodb.get_data_archive_dir())


You can also verify and print the default DISDRODB Metadata Archive and Data Archive directories by typing the following command in the terminal:

.. code:: bash

   disdrodb_data_archive_directory
   disdrodb_metadata_archive_directory

See :func:`disdrodb.get_metadata_archive_dir` and :func:`disdrodb.get_data_archive_dir` for API details.


Although not recommended for beginner users, you also have the option to define the DISDRODB Data and Metadata Archive directories using environment variables.
This can be done by setting the ``DISDRODB_DATA_ARCHIVE_DIR`` and ``DISDRODB_METADATA_ARCHIVE_DIR`` variables either directly in your terminal or by adding them to your
``.bashrc`` (or equivalent shell configuration) script.
To set them in the terminal, you can use the following commands:

.. code:: bash

   export DISDRODB_DATA_ARCHIVE_DIR="<path_of_choice_to_the_local_data_archive>/DISDRODB"
   export DISDRODB_METADATA_ARCHIVE_DIR="<path_to>/DISDRODB-METADATA/DISDRODB"

.. note:: It is important to remember that the environment variables ``DISDRODB_DATA_ARCHIVE_DIR`` and ``DISDRODB_METADATA_ARCHIVE_DIR``, if defined,
   will take priority over the default path defined in the ``.config_disdrodb.yml`` file.


3. Download Raw Disdrometer Data
-------------------------------------------

The DISDRODB Decentralized Data Archive stores raw disdrometer data contributed by the community.
Currently, a growing subset of stations is available for download.

**Check Available Stations**

Use :func:`disdrodb.available_stations` to list stations with downloadable data:

.. code:: python

    import disdrodb

    disdrodb.available_stations(available_data=True)


By periodically updating the DISDRODB Metadata Archive (``git pull``),
you can access newly available stations.

**Download Station Data**

To download raw data for specific stations:

.. code:: bash

   disdrodb_download_archive --data_sources <data_source> --campaign_names <campaign_name> --station_names <station_name> --force false

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

**Tutorial Example**

For this quick start, download a single station:

.. code:: bash

   disdrodb_download_station EPFL HYMEX_LTE_SOP3 10

where ``EPFL`` is the data source, ``HYMEX_LTE_SOP3`` is the campaign name, and ``10`` is the station name.

See :func:`disdrodb.download_station` for Python API.


4. Generate DISDRODB Products
----------------------------------------------

Once the data are downloaded, we can start the generation of the DISDRODB L0, L1 and L2E products.

The DISDRODB L0 processing chain converts raw data into a standardized format,
saving each day's data in a NetCDF file.

The DISDRODB L1 processing chain ingests the L0C product files, aggregates data at user-defined temporal resolutions,
performs quality checks, data homogenization, and apply an hydrometeor classification algorithm that allows to
differentiate between rain, snow, mixed, and non-hydrometeor particles detected by the sensors.

The DISDRODB L1 product forms the basis for generating DISDRODB L2 products, which provide advanced retrievals of drop size distribution moments
and other microphysical parameters at user-defined temporal resolutions.

**Processing Chain Overview**

.. code-block:: text

    Raw Data → L0A → L0B → L0C → L1 → L2E → L2M

- **L0**: Standardized format with quality control (see :ref:`L0A <disdrodb_l0a>`, :ref:`L0B <disdrodb_l0b>`, :ref:`L0C <disdrodb_l0c>`)
- **L1**: Temporally resampled with hydrometeor classification (see :ref:`L1 <disdrodb_l1>`)
- **L2E**: Empirical rainfall parameters and radar observables (see :ref:`L2E <disdrodb_l2e>`)
- **L2M**: Modeled DSD parameters from parametric fitting (see :ref:`L2M <disdrodb_l2m>`)

To know more about the various DISDRODB products and how to customize the processing,
please refer to the :ref:`DISDRODB Products <products>` section.
For configuration options, see :ref:`Products Configuration <products_configuration>`.

**Quick Processing Commands**

Generate L0 and L1 products with debugging enabled (processes only 3 files):

Generating DISDRODB L0 and L1 products requires only two simple commands:

.. code:: bash

   disdrodb_run_l0_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True
   disdrodb_run_l1_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True

For illustrative purposes, this code snippet here above processes just 3 raw files (``--debugging_mode True``)
with disabled parallelism (``--parallel False``) and verbose output  (``--verbose True``).

**Process All Station Data**

To process all data for a given station use:

.. code:: bash

   disdrodb_run_l0_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True
   disdrodb_run_l1_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True
   disdrodb_run_l2e_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True

See :func:`disdrodb.run_l0_station`, :func:`disdrodb.run_l1_station`, and :func:`disdrodb.run_l2e_station` for Python API.

**Create all products with Single Command**

Generate all DISDRODB products (L0 → L1 → L2E) in one command:

.. code:: bash

   disdrodb_run_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True

See :func:`disdrodb.run_station` for Python API.

**Monitor Processing**

- **Dask Dashboard**: Monitor parallel processing at http://localhost:8787/status
- **Processing Logs**: Check detailed logs for each file:

  .. code:: bash

    disdrodb_open_logs_directory EPFL HYMEX_LTE_SOP3 10

**Command Options**

- ``--force True``: Overwrite existing products
- ``--parallel True``: Enable parallel processing (default)
- ``--verbose True``: Print detailed processing information
- ``--debugging_mode True``: Process only 3 files for testing

For complete processing options, see :ref:`Archive Processing <processing>`.


6. Create Summary Figures and Table
---------------------------------------------

After generating the products, the disdrodb software provide the opportunity to automatically
generate summary figures and tables.

.. code:: bash

    disdrodb_create_summary_station EPFL HYMEX_LTE_SOP3 10

You can open the summary directory with the following command:

.. code:: bash

    disdrodb_open_product_directory SUMMARY HYMEX_LTE_SOP3 10


7. Open and Analyze DISDRODB L0 Products
---------------------------------------------------

Use :func:`disdrodb.open_dataset` to lazily open all station files for a DISDRODB product
as an ``xarray.Dataset`` (or ``pandas.DataFrame`` for L0A).

**Open All Station Files**

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

**List and Open Individual Files**

Use :func:`disdrodb.find_files` to list all files, then open individually:

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


8. Open and Analyze the DISDRODB L1 Product
---------------------------------------------------

L1 products require specifying the temporal resolution (e.g., ``1MIN``, ``5MIN``, ``10MIN``).

**Open L1 Dataset**

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


The DISDRODB L1 product includes hydrometeor classification variables for differentiating between
rain, snow, mixed precipitation, and non-hydrometeor particles.

For details on classification methodology, see :ref:`L1 Product <disdrodb_l1>`.

Here below we provide an example on how to subset and analyze the dataset by precipitation type.

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


9. Open and Analyze the DISDRODB L2E Product
---------------------------------------------------

L2E products provide empirical rainfall parameters and radar observables.

**Open L2E Dataset**

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

**L2E Product Contents**

The L2E product focuses on rainfall observations and includes:

- Drop size distribution (DSD) spectra and concentration
- DSD moments
- Microphysical parameters (rain rate, liquid water content, etc.)
- Simulated polarimetric radar variables at multiple frequencies
- Quality control flags

For details, see :ref:`L2E Product <disdrodb_l2e>` and :ref:`Radar Variable Simulations <disdrodb_radar>`.

**Example Analysis**

.. code:: python

    # Compute and load dataset
    ds = ds.compute()

    # Analyze rain rate time series
    ds["R"].plot()

    # Check radar reflectivity at C-band
    ds["ZH"].sel(frequency=5.6, method="nearest").plot()

    # Analyze DSD moments
    ds[["M3", "M4", "M6"]].to_dataframe().describe()

10. Explore Available Stations
------------------------------------

Use :func:`disdrodb.available_stations` to list stations registered in the DISDRODB Archive.

**List All Known Stations**

By default, the function returns all known stations, regardless of whether their raw data are currently available for download.
Setting ``available_data=True`` restricts the output to stations whose raw data are already available in the DISDRODB Decentralized Data Archive.
Note that some contributors have not yet made their data publicly available.

.. code:: python

    import disdrodb

    disdrodb.available_stations()  # available_data=False by default
    disdrodb.available_stations(available_data=True)


**Filter by Sensor and Measurement Interval**

.. code:: python

    import disdrodb

    disdrodb.available_stations(sensor_name="PWS100", available_data=True)
    disdrodb.available_stations(sensor_name="LPM", available_data=True)
    disdrodb.available_stations(sensor_name="PARSIVEL", measurement_interval=10, available_data=True)
    disdrodb.available_stations(sensor_name=["PARSIVEL", "PARSIVEL2"], measurement_interval=60, available_data=True)
    disdrodb.available_stations(sensor_name=["RD80"], measurement_interval=10, available_data=True)

**Filter by Station Identifiers**

Stations can be filtered by data source, campaign name, or station name.
Combine multiple filters for precise selection:

.. code:: python

    import disdrodb

    disdrodb.available_stations(data_sources=["ITALY", "EPFL"])
    disdrodb.available_stations(campaign_names="RELAMPAGO")
    disdrodb.available_stations(station_names=["TC-TO", "TC-AQ"], measurement_interval=60)


**List Processed Stations**

After generating products locally, list available processed stations with:

.. code:: python

    import disdrodb

    disdrodb.available_stations(product="L1", temporal_resolution="1MIN")
    disdrodb.available_stations(sensor_name="PARSIVEL2", product="L1", temporal_resolution="1MIN")
    disdrodb.available_stations(sensor_name="PARSIVEL2", product="L2E", temporal_resolution="1MIN")


Please note that the ``temporal_resolution`` argument is mandatory when listing DISDRODB L1 and L2 products.


11. What's Next?
---------------------

Now that you've completed the quick start, explore these topics to deepen your DISDRODB skills:

**Learn More About Products**

- :ref:`Products <products>`: Detailed descriptions of each DISDRODB product level
- :ref:`Products Configuration <products_configuration>`: Customize processing parameters, archive strategies, and model fitting
- :ref:`Radar Variable Simulations <disdrodb_radar>`: Configure multi-frequency radar simulations

**Process Your Data**

- :ref:`Archive Processing <processing>`: Batch process multiple stations and campaigns
- :ref:`Near-Real-Time Processing <nrt_processing>`: Process individual files for operational applications
- :ref:`Multi-Frequency Radar Tutorial <advanced_tutorials>`: Compute radar variables across frequency ranges

**Contribute to DISDRODB**

- :ref:`Contribute Data <contribute_data>`: Add your disdrometer stations to the Decentralized Data Archive
- :ref:`Contribute Code <contributor_guidelines>`: Develop new readers, improve processing, or add features

**Advanced Workflows**

- **L2M Products**: Fit parametric DSD models with grid search, maximum likelihood, or method of moments
  (see :ref:`L2M Product <disdrodb_l2m>` and :func:`disdrodb.run_l2m_station`)

- **Custom Processing**: Use Python API for fine-grained control over processing steps
  (see :func:`disdrodb.generate_l0a`, :func:`disdrodb.generate_l1`, :func:`disdrodb.generate_l2e`, :func:`disdrodb.generate_l2m`)

- **Data Analysis**: Use DISDRODB's xarray accessor for specialized visualizations and computations

**API Reference**

- :func:`disdrodb.open_dataset`: Open products as xarray datasets
- :func:`disdrodb.find_files`: List available product files
- :func:`disdrodb.available_stations`: Explore station catalog
- :func:`disdrodb.download_station`: Download raw data programmatically
- :func:`disdrodb.run_station`: Process complete chain for single station

**Get Help**

- `GitHub Issues <https://github.com/ltelab/disdrodb/issues>`_: Report bugs or request features
- `Discussions <https://github.com/ltelab/disdrodb/discussions>`_: Ask questions and share ideas
- `Documentation <https://disdrodb.readthedocs.io>`_: Comprehensive guides and API reference

**Stay Updated**

- Star the `DISDRODB repository <https://github.com/ltelab/disdrodb>`_ to follow development
- Update your Metadata Archive regularly: ``cd DISDRODB-METADATA && git pull``
- Join the community to stay informed about new stations and features

.. warning::

    Users are expected to properly acknowledge the data they use by citing
    and referencing each station. The corresponding references, recommended
    citations, and DOIs are available in the DISDRODB netCDFs/xarray.Dataset
    global attributes, as well as in the DISDRODB Metadata Archive.
