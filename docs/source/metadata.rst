
=========================
Metadata
=========================

The metadata for each station are defined in a YAML file.
The metadata YAML file expects a standardized set of keys.

There are 7 metadata keys for which is mandatory to specify the value :

* the ``data_source`` must be the same as the data_source where the metadata are located.
* the ``campaign_name`` must be the same as the campaign_name where the metadata are located.
* the ``station_name`` must be the same as the name of the metadata YAML file without the .yml extension.
* the ``sensor_name`` must be one of the implemented sensor configurations. See ``disdrodb.available_sensor_names()``.
  If the sensor which produced your data is not within the available sensors, you first need to add the sensor
  configurations. For this task, read the section `Add new sensor configs <https://disdrodb.readthedocs.io/en/latest/sensor_configs.html>`__.
* the ``raw_data_format`` must be either ``'txt'`` or ``'netcdf'``. ``'txt'`` if the source data are text/ASCII files. ``'netcdf'`` if source data are netCDFs.
* the ``platform_type`` must be either ``'fixed'`` or ``'mobile'``. If ``'mobile'``, the DISDRODB L0 processing accepts latitude/longitude/altitude coordinates to vary with time.
* the ``reader`` name is essential to enable to select the correct reader when processing the station.

.. note::
    The **reader** key value must be defined with the pattern ``<READER_DATA_SOURCE>/<READER_NAME>``:

    - ``<READER_DATA_SOURCE>`` is the parent directory within the disdrodb software where the reader is defined. Typically it coincides with the ``<DATA_SOURCE>`` of the DISDRODB archive.

    - ``<READER_NAME>`` is the name of the python file where the reader is defined.


    For example, to use the GPM IFLOODS reader (defined at `disdrodb.l0.reader.GPM.IFLOODS.py <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers/GPM/IFLOODS.py>`_)
    to process the data, you specify the reader name ``GPM/IFLOODS``.


To check the validity of the metadata YAML files, run the following code:

.. code-block:: python

    from disdrodb import check_archive_metadata_compliance, check_archive_metadata_geolocation

    base_dir = "<...>/DISDRODB"
    check_archive_metadata_compliance(base_dir)
    check_archive_metadata_geolocation(base_dir)




The list of the standard metadata keys and their description is provided here below:


.. csv-table:: Mandatory keys
   :align: left
   :file: ./metadata_csv/Mandatory_Keys.csv
   :widths: auto
   :header-rows: 1


.. csv-table:: Station description
   :align: left
   :file: ./metadata_csv/Description.csv
   :widths: auto
   :header-rows: 1


.. csv-table:: Deployment info
   :align: left
   :file: ./metadata_csv/Deployment_Info.csv
   :widths: auto
   :header-rows: 1


.. csv-table:: Sensor Info
   :align: left
   :file: ./metadata_csv/Sensor_Info.csv
   :widths: auto
   :header-rows: 1


.. csv-table:: Source information
   :align: left
   :file: ./metadata_csv/Source_Info.csv
   :widths: auto
   :header-rows: 1


.. csv-table:: Data Attribution
   :align: left
   :file: ./metadata_csv/Data_Attribution.csv
   :widths: auto
   :header-rows: 1
