.. _products_configuration:

============================
Products Configuration
============================

DISDRODB provides a flexible configuration system that allows full customization
of the processing chain for generating its products. Each DISDRODB product (L0C, L1, L2E, L2M)
can be configured globally or customized for specific sensors and temporal resolutions.

-------------------------------------------------
Configuration Directory Structure
-------------------------------------------------

DISDRODB products configuration files are organized in a hierarchical directory structure:

.. code-block:: text

    products_configs_dir/
    ├── L0C/
    │   ├── global.yaml              # Global L0C configuration
    │   └── <sensor_name>/           # Optional sensor-specific configurations
    │       └── global.yaml
    ├── L1/
    │   ├── global.yaml              # Global L1 configuration
    │   └── <sensor_name>/           # Optional sensor-specific configurations
    │       ├── global.yaml
    │       └── <temporal_res>.yaml  # Optional custom temporal resolution configurations
    ├── L2E/
    │   ├── global.yaml              # Global L2E configuration
        └── <temporal_res>.yaml      # Global custom temporal resolution configuration
    │   └── <sensor_name>/           # Optional sensor-specific configurations
    │       ├── global.yaml
    │       └── <temporal_res>.yaml  # Optional custom temporal resolution configurations
    └── L2M/
        ├── global.yaml              # Global L2M configuration
        └── <temporal_res>.yaml      # Global custom temporal resolution configuration
        ├── <sensor_name>/           # Optional sensor-specific configs
        │   ├── global.yaml
        │   └── <temporal_res>.yaml  # Optional custom temporal resolution configurations
        └── MODELS/                  # L2M model configurations
            ├── model1.yaml
            ├── model2.yaml
            └── ...

**Configuration Priority**

Products options are loaded with the following priority (from lowest to highest):

1. Global product configuration (``<product>/global.yaml``)
2. Sensor-specific global configuration (``<product>/<sensor_name>/global.yaml``)
3. Temporal resolution configuration (``<product>/<temporal_res>.yaml``)
4. Temporal resolution sensor-specific configuration (``<product>/<sensor_name>/<temporal_res>.yaml``)

Higher priority configurations override lower priority ones for matching keys.

You can quickly access your products configuration directory from the terminal:

.. code-block:: bash

    disdrodb_open_products_options


-------------------------------------------------
Setting Up Custom Products Configurations
-------------------------------------------------

DISDRODB comes with predefined product configurations in the ``disdrodb/etc/products/``
directory. To customize the processing chain for your data:

**1. Copy the default products configurations**

.. code-block:: python

    from disdrodb.configs import copy_default_products_configs

    products_configs_dir = "/path/to/your/custom/configs"
    copy_default_products_configs(products_configs_dir)

**2. Set the products configuration directory**

.. code-block:: python

    import disdrodb

    products_configs_dir = "/path/to/your/custom/configs"
    disdrodb.define_configs(products_configs_dir=products_configs_dir)

**3. Open the products configuration directory**

You can quickly access your configuration directory from the terminal:

.. code-block:: bash

    disdrodb_open_products_options

**4. Validate products options**

After modifying configurations, validate them:

.. code-block:: bash

    disdrodb_check_products_options

This command checks:

- Directory structure validity
- Presence of required files
- YAML syntax correctness
- Configuration option values
- Temporal resolution consistency across products
- L2M model configuration validity

-------------------------------------------------
Configuration Options
-------------------------------------------------

Archive Options
^^^^^^^^^^^^^^^^^^

Archive options control how processed data is organized into output files.

**Strategy: time_block**

Groups data into fixed time periods (e.g., daily, weekly, monthly files):

.. code-block:: yaml

    archive_options:
      strategy: time_block
      strategy_options:
        freq: day  # Options: day, week, month, year
      folder_partitioning: year/month

**Strategy: event**

Groups data by precipitation events:

.. code-block:: yaml

    archive_options:
      strategy: event
      strategy_options:
        variable: n_particles
        detection_threshold: 2            # Minimum number of particles to detect precipitation
        neighbor_min_size: 1              # Minimum neighbors to form event
        neighbor_time_interval: 1min      # Time window for neighbor search
        event_max_time_gap: 30min         # Maximum gap within event
        event_min_duration: 1min          # Minimum event duration
        event_min_size: 10                # Minimum event size (timesteps)
      folder_partitioning: year/month

Currently, only the ``time_block`` strategy is supported for DISDRODB L0C and L1 products.

**Folder Partitioning**

Controls the subdirectory structure for output files. Options include:

- ``year``: ``YYYY/``
- ``year/month``: ``YYYY/MM/``
- ``year/month/day``: ``YYYY/MM/DD/``
- ``year/week``: ``YYYY/WW/``

.. note::

   **Memory Considerations**

   DISDRODB L0C and L1 products default to daily time blocks, allowing safe processing of large
   datasets on machines with limited memory.

   DISDRODB L2E and L2M products default to monthly files since they typically compute rainfall
   parameters only for rainy timesteps, resulting in smaller data volumes.

   **Optimization Tips:**

   - On machines with sufficient memory, increase time block frequency (weekly or monthly) to reduce output file count
   - If experiencing out-of-memory errors, switch to daily archiving
   - Adjust ``DASK_NUM_WORKERS`` environment variable to control parallel processing workers


.. _config_radar:

Radar Simulation Options
^^^^^^^^^^^^^^^^^^^^^^^^^^

Available for DISDRODB L2E and L2M products when ``pytmatrix`` is installed.
For details, see :ref:`Polarimetric Radar Variables Simulations <disdrodb_radar>`.


.. code-block:: yaml

    radar_enabled: true
    radar_options:
      frequency: [5.6, 9.7, X]           # GHz or band names (S, C, X, Ku, Ka, W)
      num_points: 50                     # T-matrix integration points
      diameter_max: 8.0                  # Maximum diameter (mm)
      canting_angle_std: 10              # Canting angle std dev (degrees)
      axis_ratio_model: Thurai2007       # Axis ratio models
      permittivity_model: Turner2016   # Permittivity models
      water_temperature: 10              # Water temperature (°C)
      elevation_angle: 0                 # Elevation angle (degrees)

All radar options accept single values or lists for batch processing.
When specified as lists, each option becomes a dimension in the output DISDRODB L2E and L2M datasets,
allowing multi-frequency or multi-parameter radar simulations.

-------------------------------------------------
L0C Product Configuration
-------------------------------------------------

.. _config_l0c:

The L0C configuration controls time consistency and file consolidation.
For product description, see :ref:`DISDRODB L0C Product <disdrodb_l0c>`.

**Available Options**

.. code-block:: yaml

    archive_options:
      strategy: time_block               # Only time_block supported for L0C
      strategy_options:
        freq: day                        # Consolidation period
      folder_partitioning: year/month    # Output folder structure

**Key Notes**

- Only ``time_block`` archiving strategy is currently supported
- Daily consolidation (``freq: day``) ensures time-consistent L0B records
- No temporal resolution options available for L0C

**Example Configuration**

.. literalinclude:: ../../disdrodb/etc/products/L0C/global.yaml
   :language: yaml
   :linenos:

-------------------------------------------------
L1 Product Configuration
-------------------------------------------------

.. _config_l1:

The L1 configuration defines temporal resampling and aggregation settings.
For product description, see :ref:`DISDRODB L1 Product <disdrodb_l1>`.

**Available Options**

.. code-block:: yaml

    temporal_resolutions:              # List of resolutions to generate
      - 30S
      - 1MIN
      - 5MIN
      - ROLL10MIN                      # ROLL suffix for rolling windows

    archive_options:
      strategy: time_block             # Only time_block currently supported for L1
      strategy_options:
        freq: day
      folder_partitioning: year/month

**Temporal Resolution Syntax**

- ``<N>S``: Fixed N-second intervals (e.g., ``30S``, ``60S``)
- ``<N>MIN``: Fixed N-minute intervals (e.g., ``1MIN``, ``5MIN``, ``10MIN``)
- ``ROLL<N>MIN``: Rolling N-minute windows (e.g., ``ROLL5MIN``, ``ROLL10MIN``)

Rolling windows use sliding windows for aggregation, reducing "data loss"
compared to fixed-interval resampling.

**Key Notes**

- All temporal resolutions are generated in a single processing run
- Rolling windows reduce data "loss" during resampling
- DISDRODB L1 processing chain only supports ``time_block`` strategy currently

**Example Configuration**

.. literalinclude:: ../../disdrodb/etc/products/L1/global.yaml
   :language: yaml
   :linenos:

.. _config_l2e:

-------------------------------------------------
L2E Product Configuration
-------------------------------------------------

The L2E configuration controls filtering drop spectra.
For product description, see :ref:`DISDRODB L2E Product <disdrodb_l2e>`.

**Available Options**

.. code-block:: yaml

    temporal_resolutions:
      - 1MIN
      - 5MIN

    archive_options:
      strategy: time_block                  # time_block or event
      strategy_options:
        # ... (see Archive Options above)
      folder_partitioning: year/month

    product_options:
      # Computation flags
      compute_spectra: false           # Compute mass spectra, rainfall spectra
      compute_percentage_contribution: false

      # Quality control thresholds
      minimum_ndrops: 10               # Minimum drops required
      minimum_nbins: 3                 # Minimum populated bins
      minimum_rain_rate: 0.1           # Minimum rain rate (mm/h)

      # Spectrum filtering
      fall_velocity_model: Beard1976   # Fall velocity model
      minimum_diameter: 0.0            # Diameter filter (mm)
      maximum_diameter: 10.0
      minimum_velocity: 0.0            # Velocity filter (m/s)
      maximum_velocity: 12.0

      # Drop filtering options
      keep_mixed_precipitation: false  # Keep mixed phase
      above_velocity_fraction: null    # Fraction threshold above terminal velocity
      above_velocity_tolerance: 0.5    # Tolerance above terminal velocity (m/s)
      below_velocity_fraction: null    # Fraction threshold below terminal velocity
      below_velocity_tolerance: 0.5    # Tolerance below terminal velocity (m/s)
      maintain_drops_smaller_than: 1.0 # Keep drops < threshold (mm) if maintain_smallest_drops true
      maintain_drops_slower_than: 1.0  # Keep drops < threshold (m/s) if maintain_smallest_drops true
      maintain_smallest_drops: true    # Always keep smallest drops
      remove_splashing_drops: true     # Remove splashing drops

    radar_enabled: true
    radar_options:
      # ... (see Radar Simulation Options above)

**Key Notes**

- Currently processes only liquid precipitation
- Both ``time_block`` and ``event`` archiving strategies supported
- Raw spectra filtering to remove unrealistic rain drops based on diameter-velocity criteria
- Radar simulations require ``pytmatrix`` installation

For implementation details, see :func:`disdrodb.generate_l2e`.

**Example Configuration**

.. literalinclude:: ../../disdrodb/etc/products/L2E/global.yaml
   :language: yaml
   :linenos:

.. _config_l2m:

-------------------------------------------------
L2M Product Configuration
-------------------------------------------------

The L2M configuration controls parametric DSD model fitting.
For product description, see :ref:`DISDRODB L2M Product <disdrodb_l2m>`.

**Available Options**

.. code-block:: yaml

    models:                            # List of models to fit
      - GAMMA_GS_ND_SSE
      - GAMMA_ML

    temporal_resolutions:
      - 1MIN
      - 5MIN

    archive_options:
      strategy: event                  # time_block or event
      strategy_options:
        # ... (see Archive Options above)
      folder_partitioning: year/month

    product_options:
      fall_velocity_model: Beard1976   # Fall velocity model
      diameter_min: 0.1                # Diameter range for model fitting (mm)
      diameter_max: 8.0
      diameter_spacing: 0.1            # Grid spacing for numerical integration (mm)
      gof_metrics: true                # Compute goodness-of-fit metrics (R2, RMSE, etc.)
      minimum_ndrops: 10               # Minimum drops required for fitting
      minimum_nbins: 3                 # Minimum populated bins for fitting
      minimum_rain_rate: 0.1           # Minimum rain rate threshold (mm/h)

    radar_enabled: true
    radar_options:
      # ... (see Radar Simulation Options above)

For implementation details, see :func:`disdrodb.generate_l2m`.

**Example DISDRODB L2M default configuration**

.. literalinclude:: ../../disdrodb/etc/products/L2M/global.yaml
   :language: yaml
   :linenos:

.. _config_l2m_models:

-------------------------------------------------
L2M Model Configuration
-------------------------------------------------

Each model specified in the ``models`` list of the ``L2M/global.yaml`` file must have a corresponding YAML file
in the ``L2M/MODELS/`` directory.

DISDRODB L2M supports fitting multiple parametric drop size distribution (DSD) models using different
optimization methods. Models are fit independently, allowing comparison of different parameterizations.

**Available PSD Models**

- :class:`LognormalPSD <disdrodb.psd.models.LognormalPSD>`
- :class:`ExponentialPSD <disdrodb.psd.models.ExponentialPSD>`
- :class:`GammaPSD <disdrodb.psd.models.GammaPSD>`
- :class:`GeneralizedGammaPSD <disdrodb.psd.models.GeneralizedGammaPSD>`
- :class:`NormalizedGammaPSD <disdrodb.psd.models.NormalizedGammaPSD>`
- :class:`NormalizedGeneralizedGammaPSD <disdrodb.psd.models.NormalizedGeneralizedGammaPSD>`

**Available Optimization Methods**

- ``GS``: Grid search minimizing specified loss functions over parameter space.
  Supports single or multi-objective optimization with various error metrics.
  See :func:`get_gs_parameters <disdrodb.psd.fitting.get_gs_parameters>` for details.

- ``ML``: Maximum likelihood estimation using scipy optimizers.
  See  :func:`get_gs_parameters <disdrodb.psd.fitting.get_ml_parameters>` for details.

- ``MOM``: Method of moments.
  Analytically solves for parameters using moment relationships.
  See  :func:`get_mom_parameters <disdrodb.psd.fitting.get_mom_parameters>` for details.

**Grid Search (GS) Example**

Grid search explores the parameter space to find values that minimize the specified loss function(s).

.. code-block:: yaml

    # MODELS/GAMMA_GS_ND_SSE.yaml
    psd_model: "GammaPSD"
    optimization: "GS"
    optimization_settings:
      objectives:
        - target: "N(D)"              # Target variable: N(D), H(x), Z, R, LWC, or M<p>
          transformation: "identity"  # Transform: identity, log, sqrt
          censoring: "none"           # Censoring: none, left, right, both
          loss: "SSE"                 # Error metric: SSE, SAE, MAE, MSE, RMSE, KLDiv, WD, etc.

Multiple objectives can be combined for multi-objective optimization, allowing simultaneous
fitting to both distribution shape and integral properties:

.. code-block:: yaml

    optimization_settings:
      objectives:
        - target: "N(D)"              # Fit drop size distribution shape
          transformation: "identity"
          censoring: "none"
          loss: "SSE"
          loss_weight: 0.8            # Higher weight prioritizes distribution fit
        - target: "Z"                 # Also match radar reflectivity
          transformation: "identity"
          loss: "AE"                  # Absolute error for integral target
          loss_weight: 0.2            # Lower weight constrains integral property

**Maximum Likelihood (ML) Example**

Maximum likelihood estimation finds parameters that maximize the probability of observing
the measured drop counts.

.. code-block:: yaml

    # MODELS/GAMMA_ML.yaml
    psd_model: "GammaPSD"
    optimization: "ML"
    optimization_settings:
      init_method: "None"              # Parameter initialization: None or "MOM"
      probability_method: "cdf"        # Probability calculation: cdf or pdf
      likelihood: "multinomial"        # Likelihood function: multinomial or poisson
      truncated_likelihood: True       # Account for sensor bin truncation
      optimizer: "Nelder-Mead"         # Scipy optimizer: Nelder-Mead, Powell, BFGS, etc.

**Method of Moments (MOM) Example**

Method of moments uses analytical relationships between distribution parameters and integral moments
for fast parameter estimation.

.. code-block:: yaml

    # MODELS/GAMMA_MOM.yaml
    psd_model: "GammaPSD"
    optimization: "MOM"
    optimization_settings:
      mom_method: "M246"                # Moment combination (varies by model)
                                        # GammaPSD: M234, M246, M346, M456
                                        # LognormalPSD: M346
                                        # ExponentialPSD: M234


**Key Notes**

- Multiple models can be fit simultaneously for comparison
- Both ``time_block`` and ``event`` archiving strategies supported
- Goodness-of-fit metrics (R2, RMSE, bias) help assess model performance
- Model configurations are validated at startup to catch errors early
- Different optimization methods may yield different results; compare multiple approaches


-------------------------------------------------
Sensor-Specific Configurations
-------------------------------------------------

Sensor-specific configurations override global settings for particular sensor types.
This is useful when different sensors require different processing parameters.

**Creating sensor-specific configurations**

Create a subdirectory named after the sensor within the product directory:

.. code-block:: text

    L2E/
    ├── global.yaml          # Applied to all sensors
    └── PARSIVEL/            # Specific to OTT Parsivel
        ├── global.yaml      # Override L2E defaults for this sensor
        └── 1MIN.yaml        # Override 1min temporal resolution settings

**Example Use Cases**

- Different diameter/velocity filtering for different sensor types
- Sensor-specific quality control thresholds
- Different archive strategies per sensor
- Custom radar simulation parameters

-------------------------------------------------
Temporal Resolution-Specific Configurations
-------------------------------------------------

For DISDRODB L1, L2E, and L2M products, you can customize settings for individual temporal
resolutions. This allows different processing parameters at different time scales.

**Creating temporal resolution configurations**

Create a YAML file named after the temporal resolution:

.. code-block:: text

    L2E/
    ├── global.yaml          # Applied to all temporal resolutions
    ├── 1MIN.yaml            # Override settings for 1-minute resolution
    └── 5MIN.yaml            # Override settings for 5-minute resolution

This allow you to set different quality control thresholds, filtering options, or archive strategies for different temporal resolutions:

.. code-block:: yaml

    # L2E/1MIN.yaml
    product_options:
      minimum_ndrops: 5      # More lenient for 1-minute data
      minimum_nbins: 2

.. code-block:: yaml

    # L2E/5MIN.yaml
    product_options:
      minimum_ndrops: 20     # Stricter for 5-minute data
      minimum_nbins: 5

You only need to specify options you want to override. Non-specified options
inherit from the global configuration !


-------------------------------------------------
Configuration Validation
-------------------------------------------------

DISDRODB validates all configuration files to ensure consistency and correctness.

**Validation components**

1. **Directory structure**

   - Required product directories exist (``L0C/``, ``L1/``, ``L2E/``, ``L2M/``)
   - ``global.yaml`` file present in each product directory
   - ``L2M/MODELS/`` directory exists with model YAML files
   - Sensor subdirectory names match valid DISDRODB sensor names

2. **Configuration options**

   - Temporal resolutions use valid syntax
   - Archive strategy and options are compatible
   - Folder partitioning uses valid patterns
   - Numeric thresholds are within valid ranges
   - Radar simulation parameters are valid

3. **Model configurations**

   - PSD model names are valid
   - Optimization methods are supported
   - Optimization settings match the chosen method
   - Required parameters are present

4. **Temporal resolution consistency**

   - L1 temporal resolutions include all L2E and L2M resolutions
   - L2E temporal resolutions include all L2M resolutions

**Running validation**

.. code-block:: bash

    disdrodb_check_products_options

**Common validation errors**

- Missing ``global.yaml`` in product directory
- Invalid sensor name in subdirectory
- Temporal resolution not listed in L1 but used in L2E/L2M
- Model file missing for model listed in L2M configuration
- Invalid PSD model or optimization method in model configuration

-------------------------------------------------
Best Practices
-------------------------------------------------

1. **Start with the defaults product configuration**

   Copy default configurations and modify incrementally rather than creating
   from scratch.

2. **Validate frequently**

   Run validation after each significant change to catch errors early.

3. **Use sensor-specific configuration sparingly**

   Only create sensor-specific configurations when truly necessary. Too many
   customizations make maintenance difficult.

4. **Document customizations**

   Add comments in YAML files explaining why specific values were chosen,
   especially when deviating from defaults.

5. **Test with representative data**

   After configuration changes, test with a small representative dataset before
   processing large archives.

6. **Temporal resolution hierarchy**

   Ensure L1 includes all temporal resolutions needed by L2E and L2M to avoid
   runtime errors.

7. **Event strategy tuning**

   When using event-based archiving, tune parameters iteratively using test data
   to achieve desired event identification.
