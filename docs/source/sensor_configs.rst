.. _sensor_configurations:

=========================
Sensors Configurations
=========================

DISDRODB tailors processing of disdrometer measurements based on instrument type and characteristics.

Currently, disdrodb can process data from the
OTT Parsivel (``PARSIVEL``), OTT Parsivel2 (``PARSIVEL2``),
Thies Laser Precipitation Monitor (``LPM``) and Disdromet RD-80 (``RD80``) disdrometers.


The sensor configurations already implemented can be listed by typing the command:

.. code-block:: python

    import disdrodb

    disdrodb.available_sensor_names()


The sensor configurations are stored within the disdrodb software
`disdrodb.l0.configs <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/configs>`_ directory.
In this directory, the name of the subdirectories correspond to the ``sensor_name``.

For each sensor, the following list of configuration YAML files are required:

|   📁 disdrodb/
|   ├── 📁 l0 : Contains the software to produce the DISDRODB L0 products
|       ├── 📁 configs : Contains the specifications of various types of disdrometers
|           ├── 📁 *<sensor_name>* : e.g. PARSIVEL, PARSIVEL2, LPM, RD80
|               ├── 📜 \*.yml  : YAML files defining sensor characteristics (e.g. diameter and velocity bins)
|               ├── 📜 bins_diameter.yml : Information related to sensor diameter bins
|               ├── 📜 bins_velocity.yml : Information related to sensor velocity bins
|               ├── 📜 raw_data_format.yml : Information related to the variables logged by the sensor
|               ├── 📜 l0a_encodings.yml : Variables encodings for the L0A product
|               ├── 📜 l0b_encodings.yml : Variables encodings for the L0B product
|               ├── 📜 l0b_cf_attrs.yml : Variables CF attributes for the L0B product


To add a new sensor configuration, copy the YAML files from an existing sensor
and adapt them to your specifications.

Once you have added a new sensor configuration, validate it using:

.. code-block:: python

    from disdrodb.l0.check_configs import check_sensor_configs

    sensor_name = "PARSIVEL"  # Change with your sensor_name
    check_sensor_configs(sensor_name)

Below is detailed information about each configuration YAML file.


Sensor diameter bins
--------------------

The ``bins_diameter.yml`` file specifies drop diameter bins.
For each bin, define the ``center``, ``width``, and lower and upper ``bounds``.

Sensor velocity bins
--------------------

The ``bins_velocity.yml`` file specifies drop fall velocity bins.
For each bin, define the ``center``, ``width``, and lower and upper ``bounds``.
If the sensor (e.g., an impact disdrometer) does not measure fall velocity,
leave this file empty.

Sensor logged variables
-----------------------

The ``raw_data_format.yml`` file contains numeric information about each sensor variable.
For each variable, specify:

    * ``n_digits``: number of digits logged (including the sign, if any, but excluding the decimal point)
    * ``n_characters``: total characters (digits, decimal point, and sign)
    * ``n_decimals``: decimal digits (right of the point)
    * ``n_naturals``: natural digits (left of the point)
    * ``data_range``: valid data range
    * ``nan_flags``: values indicating missing data
    * ``field_number``: field number in the documentation

The ``null`` value should be added for character variables or when the value can not be specified.

During the DISDRODB L0 processing:

* the ``data_range``, if specified, will be used to set invalid values to ``NaN``
* the ``nan_flags`` values, if specified, will be converted to ``NaN``

The ``n_digits``, ``n_characters``, ``n_decimals`` and ``n_naturals`` information
is used to infer the raw files header when it is unknown.
See usage of the ``infer_column_names`` function in the
`reader_preparation.ipynb <https://github.com/ltelab/disdrodb/blob/main/tutorials/reader_preparation.ipynb>`_ Jupyter Notebook.

For variables whose values do not depend solely on time,
add two keys: ``n_values`` and ``dimension_order``.

The ``n_values`` key corresponds to the total number of variable values in the array.
For example, for the precipitation spectrum of the OTT PARSIVEL sensor,
characterized by 32 diameter and 32 velocity bins, ``n_values = 1024`` (32*32).

``dimension_order`` controls how the flattened precipitation spectrum array is reshaped into a 2D matrix.

For example, the OTT PARSIVEL logs the precipitation spectrum by first providing
the drop count in each bin diameters for the velocity bin 1, then for velocity bin 2 and so on.
The flattened array looks like ``[v1d1 ... v1d32, v2d1, ..., v2d32, ...]`` and therefore
``dimension_order = ["velocity_bin_center", "diameter_bin_center"]``

The Thies LPM logs the precipitation spectrum by first providing
the drop count in each velocity bin for the diameter bin 1, then for diameter bin 2 and so on.
The flattened array looks like ``[v1d1 ... v20d1, v1d2, ..., v20d2, ...]``
and therefore ``dimension_order = ["diameter_bin_center", "velocity_bin_center"]``


DISDRODB L0B variable attributes
--------------------------------

The ``l0b_cf_attrs.yml`` file specifies CF attributes for L0B netCDF variables.
Variables here must be a subset of those in ``raw_data_format.yml``.
Only these variables are referenced in other ``l0*.yml`` files.
For each variable, provide ``long_name``, ``units``, and ``description``.
Please read the Climate and Forecast Conventions guidelines for
`long_name <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#long-name>`_
and `units <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#units>`_
for more information.


DISDRODB L0A encodings
-----------------------

The ``l0a_encodings.yml`` file lists which variables can be saved in the
L0A Apache Parquet format and specifies each variable's data type.
Additionally, these variables are always included:

* the ``time`` column (in UTC)
* the ``latitude`` and ``longitude`` columns if the disdrometer station is mobile.


DISDRODB L0B encodings
-----------------------

The ``l0b_encodings.yml`` file lists variables saved in the L0B netCDF4 format.
For each variable, you need to specify the compression, data type,
the ``_FillValue`` for NaN-to-integer conversion, and the chunk size
across time (and diameter/velocity) dimensions.
The specified key values are used to define, for each variable, the specific netCDF4 encodings.
