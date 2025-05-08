.. _sensor_configurations:

=========================
Sensors Configurations
=========================

DISDRODB tailor the processing of the disdrometer measurements according to the instrument type and characteristics.

Currently, disdrodb enables to process data acquired from the
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


If you want to add a new sensor configuration, you will need to copy the YAML files
of one of the implemented sensors, and adapt the specifications.

Once you added a new sensor configuration, check the validity with the following command:

.. code-block:: python

    from disdrodb.l0.check_configs import check_sensor_configs

    sensor_name = "PARSIVEL"  # Change with your sensor_name
    check_sensor_configs(sensor_name)

Here below we detail further information related to each of the configuration YAML files.


Sensor diameter bins
---------------------

The ``bins_diameter.yml`` file contains the information related to the drop diameter bins.
Within the YAML file, the bins ``center``, ``width`` and lower and upper ``bounds`` must be specified.

Sensor velocity bins
---------------------

The ``bins_velocity.yml`` file contains the information related to the drop fall velocity bins.
Within the YAML file, the bins ``center``, ``width`` and lower and upper ``bounds`` must be specified.
If the sensor (i.e. impact disdrometers) does not measure the drop fall velocity, the YAML files must be defined empty!


Sensor logged variables
-------------------------

The ``raw_data_format.yml`` file contains the "numeric" information related to the variables logged by the sensor.
The following keys should be specified for each numeric variable:

    * ``n_digits``: the number of digits logged by the sensor (excluding the comma)
    * ``n_characters``: the number of characters (digits plus comma and sign)
    * ``n_decimals``: the number of decimals digits (right side of the comma)
    * ``n_naturals``: the number of natural digits (left side of the comma)
    * ``data_range``: the data range of the values logged by the sensor
    * ``nan_flags``: the value or list of values that flag ``NaN`` values
    * ``field_number``: the field number reported in the documentation

The ``null`` value should be added for character variables or when the value can not be specified.

During the DISDRODB L0 processing:

* the ``data_range``, if specified, will be used to set invalid values to ``NaN``
* the ``nan_flags`` values, if specified, will be converted to ``NaN``

The ``n_digits``, ``n_characters``, ``n_decimals`` and ``n_naturals`` information
is used to infer the raw files header when it is unknown.
See usage of the ``infer_column_names`` function in the
`reader_preparation.ipynb <https://github.com/ltelab/disdrodb/blob/main/tutorials/reader_preparation.ipynb>`_ Jupyter Notebook.

For the variables which values do not depend only on the time dimension, it is necessary
to specify 2 additional keys: ``n_values`` and ``dimension_order``

The ``n_values`` key corresponds to the total number of variable values in the array.
For example, for the precipitation spectrum of the OTT PARSIVEL sensor,
characterized by 32 diameter and 32 velocity bins, ``n_values = 1024`` (32*32).

The ``dimension_order`` controls how the precipitation spectrum counts logged by the
sensor have to be reshaped into a 2D matrix.

For example, the OTT PARSIVEL logs the precipitation spectrum by first providing
the drop count in each bin diameters for the velocity bin 1, then for velocity bin 2 and so on.
The flattened array looks like ``[v1d1 ... v1d32, v2d1, ..., v2d32, ...]`` and therefore
``dimension_order = ["velocity_bin_center", "diameter_bin_center"]``

The Thies LPM logs the precipitation spectrum by first providing
the drop count in each velocity bin for the diameter bin 1, then for diameter bin 2 and so on.
The flattened array looks like ``[v1d1 ... v20d1, v1d2, ..., v20d2, ...]``
and therefore ``dimension_order = ["diameter_bin_center", "velocity_bin_center"]``


DISDRODB L0B variables attributes
---------------------------------

The ``l0b_cf_attrs.yml`` file defines the DISDRODB L0B netCDF variable CF attributes.
The variables defined in this file must be a subset of the variables listed in the ``raw_data_format.yml`` file.
Only the variables defined in the ``l0b_cf_attrs.yml`` file are used in the other ``l0*.yml`` files.
The expected keys for each variable are: ``long_name``, ``units`` and ``description``.
Please read the Climate and Forecast Conventions guidelines for
`long_name <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#long-name>`_
and `units <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#units>`_
for more information.


DISDRODB L0A encodings
-----------------------

The ``l0a_encodings.yml`` file lists the variables that are allowed to be saved into the
DISDRODB L0A Apache Parquet format.
The file also specifies the type (i.e. integer/floating precision/string)
each variable is saved in the Apache Parquet binary format.
In addition to the specified variables, also the following variables are allowed
to be saved into the DISDRODB L0A files:

* the ``time`` column (in UTC)
* the ``latitude`` and ``longitude`` columns if the disdrometer station is mobile.


DISDRODB L0B encodings
-----------------------

The ``l0b_encodings.yml`` file lists the variables that are allowed to be saved into the
DISDRODB L0B netCDF4 format.

For each variable, you need to specify the compression options, the data type,
the ``_FillValue`` (to convert the ``NaN`` values when saving arrays to an integer data type) and the chunk size
across the time (and diameter and/or velocity) dimensions.
The specified key values are used to define, for each variable, the specific netCDF4 encodings.
