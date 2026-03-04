.. _quick_start:

=========================
Quick Start
=========================

This quick start guide demonstrates how to download raw disdrometer data from the DISDRODB
Decentralized Data Archive and generate DISDRODB products on your local machine.

**Prerequisites:**

- Virtual environment with DISDRODB installed (see :ref:`Installation <installation>`)
- DISDRODB Metadata Archive cloned locally
- Sufficient disk space for raw data and products

**What You'll Learn:**

- How to configure DISDRODB directories
- How to download station data from the archive
- How to generate DISDRODB L0, L1, and L2E products
- How to open and analyze products with xarray
- How to explore available stations and filter by criteria


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

.. note::

   The DISDRODB Metadata Archive is regularly updated with new stations and metadata.
   We recommend updating your local copy periodically by running ``git pull``
   inside the ``DISDRODB-METADATA`` directory.


2. Configure DISDRODB Directories
------------------------------------------

DISDRODB requires two main directory paths to operate:

- **Metadata Archive Directory** (``metadata_archive_dir``): Path to the ``DISDRODB`` subdirectory
  within your cloned ``DISDRODB-METADATA`` repository (contains station information and metadata)
- **Data Archive Directory** (``data_archive_dir``): Path where DISDRODB will store downloaded raw data
  and all processed products (L0, L1, L2)

DISDRODB will search for a configuration file named ``~/.config_disdrodb.yml`` in your home directory.

**Create Configuration File**

To create the DISDRODB configuration file, adapt and run the following Python code snippet.
See :func:`disdrodb.define_configs` for more details.
Note that paths must end with ``\DISDRODB`` on Windows or ``/DISDRODB`` on macOS/Linux.

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


You can also verify and print the default DISDRODB Metadata Archive and Data Archive directories
using the following terminal commands:

.. code:: bash

   disdrodb_data_archive_directory
   disdrodb_metadata_archive_directory

See :func:`disdrodb.get_metadata_archive_dir` and :func:`disdrodb.get_data_archive_dir` for API details.


Although not recommended for beginner users, you can also define the DISDRODB Data and Metadata Archive directories using environment variables.
Set the ``DISDRODB_DATA_ARCHIVE_DIR`` and ``DISDRODB_METADATA_ARCHIVE_DIR`` variables either directly in your terminal
or by adding them to your ``.bashrc`` (or equivalent shell configuration) file.

To set them in the terminal:

.. code:: bash

   export DISDRODB_DATA_ARCHIVE_DIR="<path_of_choice_to_the_local_data_archive>/DISDRODB"
   export DISDRODB_METADATA_ARCHIVE_DIR="<path_to>/DISDRODB-METADATA/DISDRODB"

.. note::

   Environment variables ``DISDRODB_DATA_ARCHIVE_DIR`` and ``DISDRODB_METADATA_ARCHIVE_DIR``,
   if defined, take priority over the paths specified in the ``.config_disdrodb.yml`` file.

**Optional: Configure T-Matrix Scattering Tables (Advanced)**

If you installed pyTmatrix to simulate radar variables in DISDRODB L2 products, you can specify a custom
directory for storing T-matrix scattering lookup tables (these tables can be large and are reused across processing runs):

.. code:: python

    import disdrodb

    scattering_table_dir = (
        "<path_of_choice_to_the_local_scattering_table_dir>/"  # Created automatically if it doesn't exist
    )
    disdrodb.define_configs(scattering_table_dir=scattering_table_dir)


3. Download Raw Disdrometer Data
-------------------------------------------

The DISDRODB Decentralized Data Archive stores raw disdrometer data contributed by the community.
A growing number of stations are currently available for download, with new stations being added regularly.

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

   disdrodb_download_archive --data_sources <data_source> --campaign_names <campaign_name> --station_names <station_name> --force False

The ``data_sources``, ``campaign_names``, and ``station_names`` parameters are optional
and allow you to restrict the download to specific data sources, campaigns, and/or stations.

**Command Parameters:**

-  ``--data_sources`` (optional): Filter by data source (e.g., institution or country name)
-  ``--campaign_names`` (optional): Filter by campaign name (measurement campaign or network)
-  ``--station_names`` (optional): Filter by station name
-  ``--force`` (optional, default = ``False``): Overwrite existing files if set to ``True``

To download data from multiple data sources, campaigns, or stations, provide a space-separated string.

For example:

* To download all EPFL and NASA data: ``--data_sources "EPFL NASA"``,

* To download stations from specific campaigns: ``--campaign_names "HYMEX_LTE_SOP3 HYMEX_LTE_SOP4"``,

* To download specific stations: ``--station_names "station_name1 station_name2"``.

**Quick Start Example**

For this tutorial, we'll download a single station from the EPFL data source:

.. code:: bash

   disdrodb_download_station EPFL HYMEX_LTE_SOP3 10

This command downloads data for:

- **Data Source**: ``EPFL`` (École Polytechnique Fédérale de Lausanne)
- **Campaign**: ``HYMEX_LTE_SOP3`` (HyMeX Long-Term Experiment Special Observation Period 3)
- **Station**: ``10`` (station identifier)

See :func:`disdrodb.download_station` for Python API.


4. Generate DISDRODB Products
----------------------------------------------

Once you have downloaded the raw data, you can generate standardized DISDRODB products.

**Understanding the Processing Chain**

DISDRODB processes raw disdrometer data through several stages:

- **L0 Processing**: Converts raw data into standardized NetCDF format with quality control.
  Each day's data is saved as a separate NetCDF file with CF-compliant metadata.

- **L1 Processing**: Ingests L0C files and aggregates data at user-defined temporal resolutions
  (e.g., 1-minute, 5-minute, 10-minute). Performs quality checks, data homogenization, and
  applies a hydrometeor classification algorithm to differentiate between rain, snow, mixed precipitation,
  and non-hydrometeor particles.

- **L2 Processing**: Generates advanced products from L1 data, including DSD moments,
  microphysical parameters (rain rate, liquid water content), and simulated radar variables.

**Processing Chain Overview**

.. code-block:: text

    Raw Data → L0A → L0B → L0C → L1 → L2E → L2M

- **L0**: Standardized format with quality control (see :ref:`L0A <disdrodb_l0a>`, :ref:`L0B <disdrodb_l0b>`, :ref:`L0C <disdrodb_l0c>`)
- **L1**: Temporally resampled with hydrometeor classification (see :ref:`L1 <disdrodb_l1>`)
- **L2E**: Empirical rainfall parameters and radar observables (see :ref:`L2E <disdrodb_l2e>`)
- **L2M**: Modeled DSD parameters from parametric fitting (see :ref:`L2M <disdrodb_l2m>`)

For detailed information about DISDRODB products and processing customization,
see the :ref:`DISDRODB Products <products>` and :ref:`Products Configuration <products_configuration>` sections.

**Quick Test Run (Debugging Mode)**

To quickly test the processing chain on a small sample, use debugging mode.
This processes only 3 raw files with verbose output:

.. code:: bash

   disdrodb_run_l0_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True
   disdrodb_run_l1_station EPFL HYMEX_LTE_SOP3 10 --debugging_mode True --parallel False --verbose True

This is useful for testing before running the full processing on all station data.

**Process All Station Data**

To process all available data for the station (recommended for actual data analysis):

.. code:: bash

   disdrodb_run_l0_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True
   disdrodb_run_l1_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True
   disdrodb_run_l2e_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True

See :func:`disdrodb.run_l0_station`, :func:`disdrodb.run_l1_station`, and :func:`disdrodb.run_l2e_station` for Python API.

**Create All Products with a Single Command**

Generate all DISDRODB products (L0 → L1 → L2E) in one command:

.. code:: bash

   disdrodb_run_station EPFL HYMEX_LTE_SOP3 10 --parallel True --force True

See :func:`disdrodb.run_station` for Python API.

**Monitor Processing**

While processing runs, you can monitor progress:

- **Dask Dashboard**: View real-time parallel processing status at http://localhost:8787/status
- **Processing Logs**: Check detailed logs for troubleshooting:

  .. code:: bash

    disdrodb_open_logs_directory EPFL HYMEX_LTE_SOP3 10

**Common Command Options**

- ``--force True``: Overwrite existing product files (useful when reprocessing)
- ``--parallel True``: Enable parallel processing across multiple CPU cores (default: ``True``)
- ``--verbose True``: Print detailed processing information to console
- ``--debugging_mode True``: Process only 3 files for quick testing (default: ``False``)

For complete processing options and batch processing, see :ref:`Archive Processing <processing>`.


5. Create Summary Figures and Tables
---------------------------------------------

After generating products, DISDRODB can automatically create summary visualizations and statistics
for your station. These summaries provide a quick overview of data availability, quality, and key measurements.

.. code:: bash

    disdrodb_create_summary_station EPFL HYMEX_LTE_SOP3 10

To view the generated summaries, open the summary directory:

.. code:: bash

    disdrodb_open_product_directory SUMMARY HYMEX_LTE_SOP3 10


6. Open and Analyze DISDRODB L0 Products
---------------------------------------------------

DISDRODB provides convenient functions to open and analyze processed products.
Use :func:`disdrodb.open_dataset` to lazily load all station files for a product
as an ``xarray.Dataset`` (or ``pandas.DataFrame`` for L0A products).

**Open All Station Files**

This approach opens all files for a station at once using lazy loading (data is only read from disk when needed):

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

Alternatively, you can list all product files and open them individually.
This is useful when you want to process files one at a time or inspect specific dates:

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


7. Open and Analyze the DISDRODB L1 Product
---------------------------------------------------

The L1 product is the recommended product for most analyses, as it provides quality-controlled,
temporally aggregated data with hydrometeor classification.

When opening L1 products, you must specify the temporal resolution (e.g., ``1MIN``, ``5MIN``, ``10MIN``).

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


**Understanding Hydrometeor Classification**

The L1 product includes classification variables to identify different precipitation types.
This allows you to filter data by hydrometeor type (rain, snow, graupel, hail) and
analysis quality (valid measurements vs. artifacts).

For details on the classification methodology, see :ref:`L1 Product <disdrodb_l1>`.

**Filter Data by Precipitation Type**

Here's how to subset and analyze the dataset by precipitation type:

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


8. Open and Analyze the DISDRODB L2E Product
---------------------------------------------------

The L2E product provides derived microphysical parameters and simulated radar variables,
making it ideal for rainfall analysis and radar intercomparison studies.

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

The L2E product focuses on rainfall observations and provides:

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

9. Explore Available Stations
------------------------------------

DISDRODB provides powerful filtering capabilities to explore the station catalog.
Use :func:`disdrodb.available_stations` to discover stations that match your criteria.

**List All Known Stations**

By default, this function returns all stations registered in the DISDRODB Metadata Archive,
regardless of whether their raw data are currently available for download.

To see only stations with downloadable data, use ``available_data=True``.
Note that some contributors have registered their stations but not yet made their data publicly available.

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

You can filter stations by data source, campaign name, or station name.
Multiple filters can be combined for precise selection:

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


.. note::

   The ``temporal_resolution`` argument is required when listing L1 and L2 products,
   as these products can exist at multiple temporal resolutions.


10. What's Next?
---------------------

Congratulations! You've completed the DISDRODB quick start tutorial.
Here are some next steps to deepen your knowledge and make the most of DISDRODB:

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

- **L2M Products**: Fit parametric DSD models (gamma, exponential, etc.) using grid search,
  maximum likelihood, or method of moments (see :ref:`L2M Product <disdrodb_l2m>` and :func:`disdrodb.run_l2m_station`)

- **Custom Processing**: Use the Python API for fine-grained control over individual processing steps
  (see :func:`disdrodb.generate_l0a`, :func:`disdrodb.generate_l1`, :func:`disdrodb.generate_l2e`, :func:`disdrodb.generate_l2m`)

- **Data Analysis**: Leverage DISDRODB's xarray accessor methods for specialized visualizations,
  event detection, and DSD computations

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

**Important: Data Citation**

.. warning::

    When using DISDRODB data in your research, you **must properly cite and acknowledge**
    each station's data source. This is essential for recognizing data contributors' efforts
    and maintaining the open science ecosystem.

    Citation information, DOIs, and recommended references are available in:

    - DISDRODB NetCDF/xarray.Dataset global attributes
    - DISDRODB Metadata YAML file of each station
