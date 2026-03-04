.. _sensor_configurations:

=========================
Sensor Configurations
=========================

DISDRODB tailors the processing of disdrometer measurements based on instrument type and characteristics.

**Currently Supported Sensors:**

DISDRODB can currently process data from:

* Distromet RD-80 (``RD80``)
* OTT Parsivel (``PARSIVEL``)
* OTT Parsivel2 (``PARSIVEL2``)
* Thies Laser Precipitation Monitor (``LPM``)
* Campbell Present Weather Sensor 100 (``PWS100``)
* Eigenbrot Optical Disdrometer 470 (``ODM470``),
* Biral Visibility and Present Weather Sensors (``SWS250``).


**List Available Sensor Configurations**

To view all available sensor configurations, use:

.. code-block:: python

    import disdrodb

    disdrodb.available_sensor_names()


**Configuration File Structure**

Sensor configurations are stored in the
`disdrodb.l0.configs <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/configs>`_ directory.
Each sensor has its own subdirectory named after the ``sensor_name``.

**Required Configuration Files**

For each sensor, the following configuration YAML files are required:

|   📁 disdrodb/
|   ├── 📁 l0 : Contains the software to produce DISDRODB L0 products
|       ├── 📁 configs : Contains specifications for various disdrometer types
|           ├── 📁 *<sensor_name>* : e.g., PARSIVEL, PARSIVEL2, LPM, RD80
|               ├── 📜 bins_diameter.yml : Diameter bin specifications
|               ├── 📜 bins_velocity.yml : Velocity bin specifications
|               ├── 📜 raw_data_format.yml : Variables logged by the sensor
|               ├── 📜 l0a_encodings.yml : Variable encodings for L0A product
|               ├── 📜 l0b_encodings.yml : Variable encodings for L0B product
|               ├── 📜 l0b_cf_attrs.yml : CF attributes for L0B product variables


**Adding a New Sensor Configuration**
To add a new sensor configuration:

1. Copy the YAML files from an existing sensor directory
2. Adapt them to your sensor's specifications
3. Validate the configuration using the code below:

.. code-block:: python

    from disdrodb.l0.check_configs import check_sensor_configs

    sensor_name = "PARSIVEL"  # Replace with your sensor_name
    check_sensor_configs(sensor_name)

**Configuration File Details**

The following sections provide detailed information about each configuration YAML file.


Sensor Diameter Bins
--------------------

The ``bins_diameter.yml`` file specifies drop diameter bins.

**Required Fields:**

For each bin, define:

* ``center``: Bin center value
* ``width``: Bin width
* ``bounds``: Lower and upper bin boundaries


Sensor Velocity Bins
--------------------

The ``bins_velocity.yml`` file specifies drop fall velocity bins.

**Required Fields:**

For each bin, define:

* ``center``: Bin center value
* ``width``: Bin width
* ``bounds``: Lower and upper bin boundaries

**Note:** If the sensor (e.g., an impact disdrometer) does not measure fall velocity,
leave this file empty.


Sensor Logged Variables
-----------------------

The ``raw_data_format.yml`` file contains numeric information about each sensor variable.

**Required Fields:**

For each variable, specify:

* ``n_digits``: Number of digits logged (including sign if present, excluding decimal point)
* ``n_characters``: Total characters (digits, decimal point, and sign)
* ``n_decimals``: Number of decimal digits (right of the decimal point)
* ``n_naturals``: Number of natural digits (left of the decimal point)
* ``data_range``: Valid data range for the variable
* ``nan_flags``: Values indicating missing or invalid data
* ``field_number``: Field number in the sensor documentation

**Note:** Use ``null`` for character variables or when a value cannot be specified.

**Usage During L0 Processing:**

During DISDRODB L0 processing:

* If ``data_range`` is specified, values outside this range are set to ``NaN``
* If ``nan_flags`` are specified, these values are converted to ``NaN``

**Header Inference:**

The ``n_digits``, ``n_characters``, ``n_decimals``, and ``n_naturals`` information
is used to infer raw file headers when they are unknown.
See the ``infer_column_names`` function usage in the
`reader_preparation.ipynb <https://github.com/ltelab/disdrodb/blob/main/tutorials/reader_preparation.ipynb>`_ Jupyter Notebook.

**Multi-Dimensional Variables:**

For variables with values that depend on more than just time (e.g., precipitation spectra),
add two additional keys:

* ``n_values``: Total number of variable values in the array
* ``dimension_order``: Order for reshaping the flattened array into a multi-dimensional matrix

**Examples:**

The ``n_values`` key corresponds to the total number of variable values in the array.

*Example 1: OTT PARSIVEL precipitation spectrum*

- 32 diameter bins × 32 velocity bins = 1024 total values
- ``n_values = 1024``

**Dimension Order:**

The ``dimension_order`` controls how the flattened precipitation spectrum array is reshaped into a 2D matrix.

*Example 1: OTT PARSIVEL*

The OTT PARSIVEL logs the precipitation spectrum by providing drop counts for each diameter bin
within velocity bin 1, then velocity bin 2, and so on.

- Flattened array: ``[v1d1 ... v1d32, v2d1, ..., v2d32, ...]``
- ``dimension_order = ["velocity_bin_center", "diameter_bin_center"]``

*Example 2: Thies LPM*

The Thies LPM logs the precipitation spectrum by providing drop counts for each velocity bin
within diameter bin 1, then diameter bin 2, and so on.

- Flattened array: ``[v1d1 ... v20d1, v1d2, ..., v20d2, ...]``
- ``dimension_order = ["diameter_bin_center", "velocity_bin_center"]``


DISDRODB L0B Variable Attributes
--------------------------------

The ``l0b_cf_attrs.yml`` file specifies Climate and Forecast (CF) convention attributes for L0B NetCDF variables.

**Important Notes:**

* Variables listed here must be a subset of those defined in ``raw_data_format.yml``
* Only variables listed here are referenced in other ``l0*.yml`` configuration files

**Required Fields:**

For each variable, provide:

* ``long_name``: Descriptive name following CF conventions
* ``units``: Units of measurement following CF conventions
* ``description``: Detailed description of the variable

**Resources:**

For more information, consult the Climate and Forecast Conventions guidelines:

* `long_name <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#long-name>`_
* `units <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#units>`_


DISDRODB L0A Encodings
-----------------------

The ``l0a_encodings.yml`` file specifies which variables are saved in the
L0A Apache Parquet format and defines each variable's data type.

**Automatically Included Variables:**

The following variables are always included in L0A products:

* ``time``: Timestamp column (in UTC)
* ``latitude`` and ``longitude``: Geolocation columns (if the disdrometer station is mobile)


DISDRODB L0B Encodings
-----------------------

The ``l0b_encodings.yml`` file specifies encodings for variables saved in the L0B NetCDF4 format.

**Required Specifications:**

For each variable, you must specify:

* **Compression**: Compression method and level
* **Data type**: NetCDF data type (e.g., float32, int16)
* **_FillValue**: Fill value used for NaN-to-integer conversion
* **Chunking**: Chunk size across time (and diameter/velocity) dimensions

These specifications are used to define the NetCDF4 encodings for each variable,
optimizing storage and access performance.
