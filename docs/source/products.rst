.. _products:

============================
Products
============================

DISDRODB organizes disdrometer observations into a set of standardized
products, generated sequentially from raw sensor output to physically
meaningful microphysical and radar-derived quantities.

Each product has a well-defined scope, input requirements, quality-control
steps, and output format. The uniform structure across all DISDRODB stations
enables reproducible analysis and consistent downstream processing.

DISDRODB products processing chain can be fully customized by users.
See the `Products Configuration <_products_configuration>`_ for details.


-------------------------------------------------
DISDRODB L0A Product
-------------------------------------------------

.. _disdrodb_l0a:

The **L0A product** is the first standardized representation of raw disdrometer
data within DISDRODB.

**Purpose**
  Convert heterogeneous raw text files into a cleaned, standardized tabular
  dataset compliant with DISDRODB conventions.

**Input**
  - Raw disdrometer text files
  - Optional issue files defining problematic time periods

**Description**
  Raw files are read using a *reader function* specified in the metadata of the station.
  The reader returns a tabular dataset where:

  - Each row corresponds to a measurement timestep
  - Each column corresponds to a logged variable
  - Column names follow DISDRODB naming conventions

  Variables representing arrays (e.g. raw particle spectra, mean fall velocity
  per diameter bin, raw drop number concentration) are stored as delimited strings.
  These are reshaped into multidimensional arrays in the L0B product.

**Quality Control and Standardization**
  - Removal of rows with missing or duplicated timestamps
  - Exclusion of periods flagged as problematic in the issues files
  - Conversion of corrupted numeric entries to NaN
  - Trimming and cleaning of string fields
  - Enforcement of data types and valid ranges

  All detected issues are logged in dedicated log files.

**Output**
  - DISDRODB-compliant L0A dataset in Apache Parquet binary format.
  - The variables varies across sensors and stations depending on
    the logged quantities. The raw spectrum variable is always included.
  - Detailed processing logs highlighting data issues and cleaning actions.

-------------------------------------------------
DISDRODB L0B Product
-------------------------------------------------

.. _disdrodb_l0b:

The **L0B product** converts tabular L0A data into the netCDF4 data model.

**Purpose**
  Provide a self-describing dataset with explicit physical dimensions and
  standardized metadata.

**Input**
  - L0A dataset

**Description**
  The L0B processing chain:

  - Parses string-encoded array variables into numerical arrays
  - Constructs an ``xarray.Dataset`` with dimensions:
    - ``time``
    - ``diameter_bin_center``,
    - ``velocity_bin_center`` (when available)
  - Adds bin centers and bounds for diameter and velocity
  - Attaches station geolocation (longitude, latitude, altitude)

**Metadata**
  - Climate and Forecast (CF) compliant variable attributes
  - Attribute Convention for Data Discovery (ACDD) global attributes
  - Optimized NetCDF encodings to reduce disk usage

**Output**
  - NetCDF4 L0B files suitable for direct scientific analysis
  - The variables varies across sensors and stations depending on
    the logged quantities. The raw spectrum variable is always included.

-------------------------------------------------
DISDRODB L0C Product
-------------------------------------------------

.. _disdrodb_l0c:

The **L0C product** ensures temporal consistency and prepares the data for
resampling and higher-level processing.
The L0C processing chain regroups potentially heterogeneous L0B files into
fixed-period files (daily by default, or weekly/monthly depending on user
configuration).

**Purpose**
  Consolidates potentially heterogeneous L0B files into fixed-period output
  Ensures files with fixed measurement intervals, unique timesteps, and
  consistent time axes.

**Description**
  The L0C chain:

  - Removes duplicated timesteps created by file concatenation
  - Discards measurements with inconsistent or unexpected measurement intervals
  - Separates data into distinct datasets if multiple measurement intervals are detected
  - Corrects small timestamp drifts to exact interval boundaries
  - Stores the verified measurement interval as a dataset coordinate

**Quality Control**
  - Computation of ``qc_time`` to assess temporal continuity
  - Logging of irregular sampling patterns and intermittent measurements

**Output**
  - Time-consistent L0C datasets
  - The variables varies across sensors and stations depending on
    the logged quantities. The raw spectrum variable is always included.

-------------------------------------------------
DISDRODB L1 Product
-------------------------------------------------

.. _disdrodb_l1:

The **L1 product**  aggregates disdrometer observations at multiple temporal resolutions (depending on user
configuration), and performs hydrometeor classification to facilitate downstream analysis and tailored product development.
Starting from DISDRODB L1 products, all stations have the same variables and data structure.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Temporal Resampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**
  Aggregate particle spectra and auxiliary variables to user-defined temporal
  resolutions.

**Features**
  - Fixed-interval and rolling-window aggregation
  - Typical resolutions: 1, 5, and 10 minutes
  - Rolling windows reduce data loss and increase sample density

**Quality Control**
  - ``qc_resampling`` reports the fraction of missing data within each aggregation
    window

**Customization**
    - User-defined temporal resolutions and aggregation methods

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Hydrometeor Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**
  Identify the dominant hydrometeor type and precipitation phase at each timestep.

**Description**
  - Operates on the diameter-velocity particle spectrum
  - Applies sensor-specific noise filtering
  - Uses physically based size-velocity masks
  - Adjusts fall-velocity relationships for air density (altitude)
  - Optionally refines classification using temperature

**Output**
  - Hydrometeor and precipitation-type labels
  - Classification-related quality-control flags

-------------------------------------------------
DISDRODB L2E Product (Empirical)
-------------------------------------------------

.. _disdrodb_l2e:

The **L2E product** derives microphysical parameters and radar observables
directly from observed particle spectra.
Currently, the DISDRODB L2E product provides geophysical quantities for rainfall observations only.

**Purpose**
  Compute integral drop size distribution (DSD) parameters and simulate
  polarimetric radar variables for rainfall.

**Description**
  - Selection of liquid precipitation timesteps
  - Diameter and fall-velocity filtering
  - Estimation of drop number concentration
  - Computation of DSD moments and rainfall variables
  - T-matrix simulation of polarimetric radar observables

**Customization**
  - User-defined thresholds on minimum particle counts, populated bins, and rain rate
  - User-defined spectrum filtering criteria

-------------------------------------------------
DISDRODB L2M Product (Modelled)
-------------------------------------------------

.. _disdrodb_l2m:

The **L2M product** fits parametric DSD models to observed drop number concentration derived in DISDRODB L2E products.
From the modeled distributions, it derives microphysical and radar variables.

**Purpose**
  Support microphysical studies, radar retrieval development, and model
  evaluation.

**Description**
  - Fits multiple parametric DSD models (e.g. lognormal, exponential, gamma, generalized gamma and normalized versions)
  - Supports grid search, maximum likelihood, and method-of-moments estimation
  - Computes goodness-of-fit diagnostics
  - Derives integral DSD parameters and radar observables from modeled DSDs

**Applications**
  - Evaluation of bulk microphysics parameterizations
  - Development and validation of radar-based DSD retrievals

-------------------------------------------------
Radar Variable Simulations
-------------------------------------------------

.. _disdrodb_radar:

Radar variables in DISDRODB are simulated using electromagnetic scattering
calculations based on the T-matrix method. pytmatrix must be installed to
enable radar simulations (see _pytmatrix_installation in the Installation Section).

**Features**
  - Compatible with L2E (empirical) and L2M (modelled) products
  - Simulation of reflectivity, attenuation, phase, and polarimetric variables
  - Flexible configuration of radar and microphysical assumptions
  - Parallelized execution with caching for efficiency

The modular design of DISDRODB allows future integration of alternative
hydrometeor scattering models.
