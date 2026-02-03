.. _products:

============================
Products
============================

DISDRODB transforms raw disdrometer data into standardized products through
a sequential processing chain, from sensor output to physically meaningful
microphysical and radar-derived quantities.

Each product has a well-defined scope, quality-control procedures, and output
format. This uniform structure across all stations enables reproducible
analysis and consistent downstream processing.

The processing chain is fully customizable. See the
:ref:`Products Configuration <products_configuration>` for more details.

.. _disdrodb_l0a:

-------------------------------------------------
DISDRODB L0A Product
-------------------------------------------------

The DISDRODB L0A product converts heterogeneous raw disdrometer files into a
standardized tabular dataset.

**Purpose**
  Transform raw text files into cleaned, DISDRODB-compliant data.

**Input**
  - Raw disdrometer text files
  - Optional issue files defining problematic time periods

**Description**
  A station-specific *reader function* (specified in metadata) processes raw
  files to produce a tabular dataset where each row is a measurement timestep
  and each column is a logged variable following DISDRODB naming conventions.

  Array variables (particle spectra, velocity-diameter distributions) are stored
  as delimited strings, later reshaped into multidimensional arrays in L0B.

**Quality Control**
  - Removes rows with missing or duplicated timestamps
  - Excludes periods flagged in issue files
  - Converts corrupted numeric entries to NaN
  - Enforces data types and valid ranges
  - Logs all detected issues

**Output**
  - L0A dataset in Apache Parquet format
  - Variables depend on sensor type; raw spectrum always included
  - Detailed processing logs

.. _disdrodb_l0b:

-------------------------------------------------
DISDRODB L0B Product
-------------------------------------------------

The DISDRODB L0B product converts tabular L0A data into the netCDF4 data model.

**Purpose**
  Provide a self-describing dataset with explicit physical dimensions and
  standardized metadata.

**Input**
  - L0A dataset

**Description**
  The L0B processing:

  - Parses string-encoded arrays into numerical arrays
  - Constructs an ``xarray.Dataset`` with dimensions: ``time``,
    ``diameter_bin_center``, and ``velocity_bin_center`` (when available)
  - Adds bin centers and bounds for diameter and velocity
  - Attaches station geolocation (longitude, latitude, altitude)

**Metadata**
  - Climate and Forecast (CF) compliant variable attributes
  - Attribute Convention for Data Discovery (ACDD) global attributes
  - Optimized NetCDF encodings to minimize disk usage

**Output**
  - NetCDF4 files suitable for scientific analysis
  - Variables depend on sensor type; raw spectrum always included

.. _disdrodb_l0c:

-------------------------------------------------
DISDRODB L0C Product
-------------------------------------------------

The DISDRODB L0C product ensures temporal consistency and consolidates L0B files
into fixed-period outputs (daily by default; configurable as weekly or monthly).

**Purpose**
  Create time-consistent datasets with fixed measurement intervals, unique
  timesteps, and standardized file grouping.

**Description**
  The L0C processing:

  - Removes duplicated timesteps from file concatenation
  - Discards measurements with inconsistent intervals
  - Separates data into distinct datasets if multiple measurement intervals exist
  - Corrects small timestamp drifts to exact interval boundaries
  - Stores the verified measurement interval as a coordinate

**Quality Control**
  - Computes ``qc_time`` to assess temporal continuity
  - Logs irregular sampling patterns and intermittent measurements

**Output**
  - Time-consistent L0C datasets grouped by fixed periods
  - Variables depend on sensor type; raw spectrum always included

For configuration options, see :ref:`DISDRODB L0C Product Configuration <config_l0c>`.

.. _disdrodb_l1:

-------------------------------------------------
DISDRODB L1 Product
-------------------------------------------------

The DISDRODB L1 product aggregates observations at multiple temporal resolutions
and performs hydrometeor classification. Starting from the DISDRODB L1 product, all stations have
the same variables and data structure. The DISDRODB L1 product serves as a common foundation
for existing and future DISDRODB L2 products.

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
  - ``qc_resampling`` reports the fraction of missing data within each window

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

For configuration options, see :ref:`DISDRODB L1 Product Configuration <config_l1>`.

.. _disdrodb_l2e:

-------------------------------------------------
DISDRODB L2E Product (Empirical)
-------------------------------------------------

The DISDRODB L2E product derives microphysical parameters and radar observables
directly from observed particle spectra. Currently, L2E provides geophysical
quantities for rainfall observations only. The default DISDRODB L2E configuration
process only timesteps with precipitation, resulting in temporally discontinuous data.

**Purpose**
  Compute integral drop size distribution (DSD) parameters and simulate
  polarimetric radar variables for rainfall.

**Description**
  - Selects liquid precipitation timesteps
  - Filters particles by diameter and fall-velocity
  - Estimates drop number concentration
  - Computes DSD moments and rainfall variables
  - Simulates polarimetric radar observables via T-matrix

**Customization**
  - User-defined thresholds on minimum particle counts, populated bins, and rain rate
  - User-defined spectrum filtering criteria

For configuration options, see :ref:`DISDRODBL2E Product Configuration <config_l2e>`.

.. _disdrodb_l2m:

-------------------------------------------------
DISDRODB L2M Product (Modelled)
-------------------------------------------------

The DISDRODB L2M product fits parametric DSD models to observed drop number
concentrations from L2E and derives microphysical and radar variables from
the fitted distributions. The default DISDRODB L2E configuration
process only timesteps with precipitation, resulting in temporally discontinuous data.

**Purpose**
  Support microphysical studies, radar retrieval development, and model
  evaluation.

**Description**
  - Fits multiple parametric DSD models (lognormal, exponential, gamma,
    generalized gamma, and normalized variants)
  - Supports grid search, maximum likelihood, and method-of-moments estimation
  - Computes goodness-of-fit diagnostics
  - Derives integral DSD parameters and radar observables from modeled DSDs

**Applications**
  - Evaluation of bulk microphysics parameterizations
  - Development and validation of radar-based DSD retrievals

For configuration options, see :ref:`DISDRODB L2M Product Configuration <config_l2m>` and
:ref:`L2M Models Configuration <config_l2m_models>`.

.. _disdrodb_radar:

-------------------------------------------------
Polarimetric Radar Variables
-------------------------------------------------

Polarimetric radar variables are simulated using electromagnetic scattering calculations
based on the T-matrix method. pytmatrix must be installed to enable radar
simulations (see :ref:`pytmatrix installation <pytmatrix_installation>`).

For configuration options, see :ref:`DISDRODB Radar Configuration Options <config_radar>`.

**Features**
  - Compatible with L2E (empirical) and L2M (modeled) products
  - Simulates reflectivity, attenuation, phase, and polarimetric variables
  - Flexible configuration of radar and microphysical assumptions
  - Parallelized execution with caching for efficiency
