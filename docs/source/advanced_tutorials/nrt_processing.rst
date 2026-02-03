.. _nrt_processing:

============================
Near-Real-Time Processing
============================

This tutorial demonstrates how to process individual files or small batches for near-real-time (NRT)
applications. Unlike batch archive processing, this workflow provides fine-grained control over each
processing step, enabling integration into operational systems, streaming pipelines, or custom workflows.

-------------------------------------------------
Overview
-------------------------------------------------

**Use Cases**

- **Operational monitoring**: Process incoming disdrometer data as it arrives
- **Real-time displays**: Update visualizations with latest observations
- **Alert systems**: Trigger warnings based on rainfall thresholds
- **Data streaming**: Integrate DISDRODB into real-time data pipelines
- **Custom workflows**: Build specialized processing chains with custom quality control

**Workflow**

This tutorial processes data through the complete DISDRODB chain manually:

.. code-block:: text

    Raw File(s) → L0A (DataFrame) → L0B (xarray) → L0C → L1 → L2E → L2M

Each step can be executed independently, allowing you to:

- Process individual files or small batches
- Skip intermediate file I/O for faster processing
- Customize quality control at each level
- Extract specific products without generating entire archive

-------------------------------------------------
Setup and Configuration
-------------------------------------------------

**Import Required Modules**

.. code-block:: python

    import numpy as np
    import xarray as xr
    import disdrodb
    from disdrodb.l0.l0c_processing import finalize_l0c_dataset
    from disdrodb.routines.options import get_product_options, get_model_options
    from disdrodb.configs import get_products_configs_dir

**Configure Station**

.. code-block:: python

    # Define station identifiers
    data_source = "EPFL"
    campaign_name = "LOCARNO_2019"
    station_name = "61"

    # Processing options
    verbose = True
    parallel = False
    temporal_resolution = "1MIN"

    # Load products configuration directory (or specify custom path)
    products_configs_dir = get_products_configs_dir()

    # Read station metadata
    metadata = disdrodb.read_station_metadata(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )


-------------------------------------------------
Step 1: Generate L0A DataFrame
-------------------------------------------------

The L0A level converts raw disdrometer files into standardized tabular format.
Here we illustrate how to manually generate the L0A DataFrame from specific raw files.

.. code-block:: python

    # Extract metadata
    sensor_name = metadata["sensor_name"]
    sample_interval = metadata["measurement_interval"]

    # Get station-specific reader
    reader = disdrodb.get_station_reader(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Define files to process (for NRT, typically the most recent file(s))
    # For illustration, here we list all existing station files and select a subset to mock NRT processing
    filepaths = disdrodb.find_files(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )
    filepaths = filepaths[10:15]

    # Process specific files
    df = disdrodb.generate_l0a(
        filepaths=filepaths,
        reader=reader,
        sensor_name=sensor_name,
        verbose=verbose,
    )

.. tip::

   **Near-Real-Time Processing**

   For operational NRT systems:

   - Monitor directory for new files using ``watchdog`` or similar tools
   - Implement error handling for incomplete or corrupted files

-------------------------------------------------
Step 2: Generate L0B Dataset
-------------------------------------------------

Convert the L0A DataFrame into an xarray Dataset with proper dimensions and metadata.

.. code-block:: python

    # Create L0B dataset
    ds_l0b = disdrodb.generate_l0b(df=df, metadata=metadata, verbose=verbose)

**What L0B Processing Does:**

- Parses string-encoded arrays into numerical arrays
- Constructs xarray Dataset with dimensions (time, diameter_bin_center, velocity_bin_center)
- Adds bin centers and bounds
- Attaches CF-compliant variable attributes
- Adds station geolocation metadata

See :func:`disdrodb.generate_l0b` for implementation details.

-------------------------------------------------
Step 3: Generate L0C Dataset
-------------------------------------------------

Ensure temporal consistency by regularizing timestamps and validating measurement intervals.

.. code-block:: python

    # Finalize L0C dataset
    # - Regularizes trailing seconds in timestamps
    # - Validates measurement interval consistency
    # - Computes quality control flags
    ds_l0c = finalize_l0c_dataset(
        ds_l0b,
        sensor_name=sensor_name,
        sample_interval=sample_interval,
        verbose=verbose,
    )

.. note::

   **Simplified L0C Processing**

   The full L0C archive processing additionally:

   - Consolidates multiple L0B files into fixed-period outputs (daily by default)
   - Removes duplicate timesteps across file boundaries
   - Splits datasets if measurement intervals change

   For NRT processing of individual files, ``finalize_l0c_dataset()`` provides
   the essential time consistency checks without file consolidation.

See :func:`disdrodb.l0.l0c_processing.finalize_l0c_dataset` for implementation details.

**Subset Data for Testing**

For faster iteration during development:

.. code-block:: python

    # Select subset and load into memory
    ds_l0c = ds_l0c.isel(time=slice(0, 1000))
    ds_l0c = ds_l0c.compute()

-------------------------------------------------
Step 4: Generate L1 Dataset
-------------------------------------------------

Resample data to desired temporal resolution and perform hydrometeor classification.

.. code-block:: python

    # Resample to target temporal resolution
    ds_l0c_resampled = ds_l0c.disdrodb.resample(
        temporal_resolution=temporal_resolution,
    )

    # Generate L1 product with hydrometeor classification
    ds_l1 = disdrodb.generate_l1(ds_l0c_resampled)

**Temporal Resolution Options:**

- Fixed intervals: ``"30S"``, ``"1MIN"``, ``"5MIN"``, ``"10MIN"``
- Rolling windows: ``"ROLL5MIN"``, ``"ROLL10MIN"``

See :func:`disdrodb.generate_l1` for implementation details.

.. tip::

   **Multiple Temporal Resolutions**

   For operational systems requiring multiple resolutions, loop over desired values:

   .. code-block:: python

       for temporal_resolution in ["1MIN", "5MIN", "10MIN"]:
           ds_resampled = ds_l0c.disdrodb.resample(temporal_resolution=temporal_resolution)
           ds_l1 = disdrodb.generate_l1(ds_resampled)
           # Process L1 → L2E → L2M for this resolution
           ...

-------------------------------------------------
Step 5: Generate L2E Dataset
-------------------------------------------------

Compute empirical rainfall parameters and radar observables from classified L1 data.

**Extract L2E Configuration**

.. code-block:: python

    # Load L2E processing options
    l2e_options = get_product_options(
        product="L2E",
        temporal_resolution=temporal_resolution,
        products_configs_dir=products_configs_dir,
    )

For configuration options, see :ref:`L2E Product Configuration <config_l2e>`.

**Generate L2E Product**

.. code-block:: python

    # Extract product options
    l2e_product_options = l2e_options["product_options"]

    # Adjust thresholds for NRT (optional)
    # Default configuration processes only rainy timesteps
    # For NRT monitoring, you may want all timesteps:
    l2e_product_options["minimum_rain_rate"] = 0
    l2e_product_options["minimum_ndrops"] = 0
    l2e_product_options["minimum_nbins"] = 0

    # Generate L2E dataset
    ds_l2e = disdrodb.generate_l2e(ds_l1, **l2e_product_options)

See :func:`disdrodb.l2.processing.generate_l2e` for implementation details.

**Add Radar Simulations (Optional)**

If ``pytmatrix`` is installed, compute radar observables:

.. code-block:: python

    # Extract radar simulation options
    l2e_radar_options = l2e_options["radar_options"]

    # Generate radar variables
    ds_l2e_radar = disdrodb.generate_l2_radar(ds_l2e, **l2e_radar_options)

    # Merge radar variables into L2E dataset
    ds_l2e.update(ds_l2e_radar)
    ds_l2e.attrs = ds_l2e_radar.attrs.copy()

See :ref:`Radar Variable Simulations <disdrodb_radar>` for details on radar options.

-------------------------------------------------
Step 6: Generate L2M Dataset
-------------------------------------------------

Fit parametric DSD models to derive modeled rainfall parameters.

**Extract L2M Configuration**

.. code-block:: python

    # Load L2M processing options
    l2m_options = get_product_options(
        product="L2M",
        temporal_resolution=temporal_resolution,
        products_configs_dir=products_configs_dir,
    )

    # Extract product options
    l2m_product_options = l2m_options["product_options"]

    # Adjust thresholds for NRT (optional)
    l2m_product_options["minimum_rain_rate"] = 0
    l2m_product_options["minimum_ndrops"] = 0
    l2m_product_options["minimum_nbins"] = 0

For configuration options, see :ref:`L2M Product Configuration <config_l2m>`.

**Select DSD Model**

Load configuration for specific parametric model:

.. code-block:: python

    # Load model configuration
    model_options = get_model_options(
        model_name="NGAMMA_GS_ND_SSE",  # GAMMA_ML, etc.
        products_configs_dir=products_configs_dir,
    )

Available models are defined in the ``L2M/MODELS/`` directory.
See :ref:`L2M Model Configuration <config_l2m_models>` for available models and options.

**Generate L2M Product**

.. code-block:: python

    # Generate L2M dataset with fitted model parameters
    ds_l2m = disdrodb.generate_l2m(
        ds_l2e,
        **model_options,
        **l2m_product_options,
    )

    # Extract fitted PSD parameters
    ds_l2m_parameters = ds_l2m.disdrodb.psd_parameters

See :func:`disdrodb.generate_l2m` for implementation details.

-------------------------------------------------
Further Reading
-------------------------------------------------

- :ref:`Archive Processing <processing>`: Batch processing for entire archives
- :ref:`Products Configuration <products_configuration>`: Customize processing options
- :ref:`Products <products>`: Detailed product descriptions
- :func:`disdrodb.generate_l0a`: L0A dataframe generation documentation
- :func:`disdrodb.generate_l0b`: L0B dataset generation documentation
- :func:`disdrodb.generate_l1`:  L1 dataset generation documentation
- :func:`disdrodb.generate_l2e`: L2E dataset generation documentation
- :func:`disdrodb.generate_l2m`: L2M dataset generation documentation
