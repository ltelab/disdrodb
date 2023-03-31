=========================
Add new sensor configs
=========================

DISDRODB tailor the processing of the disdrometer measurements according
to the instrument type and characteristics.

Several sensor configurations are already implemented and can be seen
by typing the command :

.. code-block:: python

    import disdrodb

    disdrodb.available_sensor_name()


The sensor configurations are stored within the disdrodb software
`disdrodb.l0.configs <https://github.com/ltelab/disdrodb/tree/main/disdrodb/L0/readers/GPM/IFLOODS.py>`_ directory.
In this directory, the name of the subdirectories correspond to ``the sensor_name``.

For each sensor, the following list of configuration YAML files are required:

| 📁 disdrodb/
| ├── 📁 l0 : Contains the software to produce the DISDRODB L0 products
|     ├── 📁 configs : Contains the specifications of various types of disdrometers
|     	├── 📁 `<sensor_name>` : e.g. OTT_Parsivel, OTT_Parsivel2, Thies_LPM, RD_80
|     		├── 📜 \*.yml  : YAML files defining sensor characteristics (e.g. diameter and velocity bins)
|     		├── 📜 bins_diameter.yml : Information related to sensor diameter bins
|     		├── 📜 bins_velocity.yml : Information related to sensor velocity bins
|     		├── 📜 L0A_encodings.yml : Variables encodings for the L0A product
|     		├── 📜 L0B_encodings.yml : Variables encodings for the L0B product
|     		├── 📜 L0_data_format.yml : Information related to the variables logged by the sensor
|     		├── 📜 variables.yml : Variables logged by the sensor
|     		├── 📜 variable_description.yml : Variables description
|     		├── 📜 variable_long_name.yml: Variables long_name
|     		├── 📜 variable_units.yml: Variables unit

If you want to add a new sensor configuration, you will need to copy the YAML files
of one of the implemented sensors, and adapt the specifications.

Once you added a new sensor configuration, check the validity with the following command:

    .. code-block:: python

        from disdrodb.l0.check_configs import check_sensor_configs

        sensor_name = "OTT_Parsivel"  # Change with your sensor_name
        check_sensor_configs(sensor_name)

Here below we details further information related to each of the configuration
YAML files


bins_diameter.yml file
~~~~~~~~~~~~~~~~~~~~~~~

This file contains the information related to the drop diameter bins.
Within the YAML file, the bins ``center``, ``width`` and lower and upper ``bounds``
must be specified.

bins_velocity.yml file
~~~~~~~~~~~~~~~~~~~~~~~

This file contains the information related to the drop fall velocity bins.
Within the YAML file, the bins ``center``, ``width`` and lower and upper ``bounds``
must be specified.
If the sensor (i.e. impact disdrometers) does not measure the drop fall velocity,
the YAML files must be defined empty !


L0A_encodings.yml file
~~~~~~~~~~~~~~~~~~~~~~~

This file list the variables that are allow to be saved into the
DISDRODB L0A Apache Parquet format.
The file also specified the type (i.e. integer/floating precision/string)
each variable is saved in the Apache Parquet binary format.
In addition to the specified variables, also the following variables are allowed
to be saved into the DISDRODB L0A files:

* the ``time`` column (in UTC format)
* the ``latitude`` and ``longitude`` columns if the disdrometer station is mobile.


L0B_encodings.yml file
~~~~~~~~~~~~~~~~~~~~~~~

This file list the variables that are allow to be saved into the
DISDRODB L0B netCDF format.

For each variable, you need to specify the compression options, the data type,
the _FillValue to store i.e. NaN values (if integer data type), the chunk size
across the time (and diameter and/or velocity) dimensions.
The specified key values are used to define, for each variable, the specific
netCDF encodings.

raw_data_format.yml file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file contains the "numeric" information related to the variables logged by the sensor.
The following keys should be specified for each numeric variable:

    * ``n_digits``: the number of digits logged by the sensor (excluding the comma)
    * ``n_characters``: the number of characters (digits plus comma and sign)
    * ``n_decimals``: the number of decimals digits (right side of the comma)
    * ``n_naturals``: the number of natural digits (left side of the comma)
    * ``data_range``: the data range of the values logged by the sensor
    * ``nan_flags``: the value or list of values that flag NaN values

The ``null`` value should be added for character variables or when the value can not be specified.

During the DISDRODB L0 processing:

* the ``data_range``, if specified, will be used to set unvalid values to NaN
* the ``nan_flags`` values, if specified, will be converted to NaN

The ``n_digits``, ``n_characters``, ``n_decimals`` and ``n_naturals`` information
is used to infer the raw files header when this is unknown.
See usage of the ``infer_column_names`` function in the
`reader_preparation.ipynb Jupyter Notebook <https://github.com/ltelab/disdrodb/tree/main/tutorial>`_.

For the variables which values do not depend only from the time dimension, it is necessary
to specify 2 additional keys: ``n_values`` and ``dimension_order``

The ``n_values`` key corresponds to the total number of the array variable values.
For example, for the precipitation spectrum of the OTT Parsivel sensor,
characterized by 32 diameter and 32 velocity bins, n_values = 1024 (32*32).

The ``dimension_order`` controls how the precipitation spectrum counts logged by the
sensor has to be reshaped into a 2D matrix.

For example, the OTT Parsivel logs the precipitation spectrum by first providing
the drop count in each bin diameters for the velocity bin 1, then for velocity bin 2 and so on.
The flattened array looks like [v1d1 ... v1d32, v2d1, ..., v2d32, ...] and therefore
``dimension_order = ["velocity_bin_center", "diameter_bin_center"]``

The Thies LPM logs the precipitation spectrum by first providing
the drop count in each velocity bin for the diameter bin 1, then for diameter bin 2 and so on.
The flattened array looks like [v1d1 ... v20d1, v1d2, ..., v20d2, ...]
and therefore ``dimension_order = ["diameter_bin_center", "velocity_bin_center"]``



variables.yml file
~~~~~~~~~~~~~~~~~~~~~~~

This file list define the standard name of the variables logged by the sensor.
Only these standard names are used in the other YAML config files.


variable_description.yml file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file contains a description for each variable.
The ``description`` will be attached as a variable attribute to the DISDRODB L0B netCDF

variable_units.yml file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file specify the measurement unit for each variable.
The ``units`` will be attached as a variable attribute to the DISDRODB L0B netCDF

variable_long_name.yml file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file specify the long_name for each variable.
The ``long_name`` will be attached as a variable attribute to the DISDRODB L0B netCDF.
See the `CF Conventions guidelines for long_name
<https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#long-name>`_
for more information.
