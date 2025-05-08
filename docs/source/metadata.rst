.. _metadata:

=========================
Station Metadata
=========================

The metadata for each station are defined in a YAML file.
The metadata YAML file expects a standardized set of keys.

There are 7 metadata keys for which it is mandatory to specify the value :

* the ``data_source`` must be the same as the data_source where the metadata are located.
* the ``campaign_name`` must be the same as the campaign_name where the metadata are located.
* the ``station_name`` must be the same as the name of the metadata YAML file without the .yml extension.
* the ``sensor_name`` must be one of the implemented sensor configurations. See ``disdrodb.available_sensor_names()``.
  If the sensor which produced your data is not within the available sensors, you first need to add the sensor
  configurations. For this task, read the section :ref:`Add new sensor configs <sensor_configurations>`.
* the ``platform_type`` must be either ``'fixed'`` or ``'mobile'``. If ``'mobile'``, the DISDRODB L0 processing accepts latitude/longitude/altitude coordinates to vary with time.
* the ``raw_data_format`` must be either ``'txt'`` or ``'netcdf'``. ``'txt'`` if the source raw data are text/ASCII files. ``'netcdf'`` if source raw data are netCDFs.
* the ``raw_data_glob_pattern`` defines which raw data files in the ``DISDRODB/RAW/<DATA_SOURCE>/<CAMPAIGN_NAME>/<STATION_NAME>/data`` directory will be ingested
  in the DISDRODB L0 processing chain.
  For instance, if every station raw files ends with ``.txt`` you can specify the glob pattern as  ``*.txt``.
  Because you're not including any path separators (``/``), this simple glob pattern will recurse through all subfolders
  (e.g. ``<year>/<month>/``) under ``data/`` and pick up every ``.txt`` file.
  If there are other ``.txt`` files in ``data/`` that you don't want to process (e.g. some geolocation information for mobile platforms or some auxiliary weather data),
  you can narrow the match by adding the filename prefix of the file you aim to process to the glob pattern (e.g. ``SPECTRUM_*.txt``).

  Finally, to restrict the search to a particular ``data/`` subdirectory, include that folder name in your pattern.
  Specifying ``"<custom>/*.txt`` will return only files directly inside the ``data/<custom>`` directory,
  while ``"<custom>/**/*.txt`` will return all files in the ``data/<custom>`` directory and all its (e.g. ``/<year>/<month>``) subdirectories.

* the ``reader`` reference tells the disdrodb software which reader function to use to correctly ingest the station's raw data files.
  Under the hood, a reader is simply a python function that knows how to read a raw data file and make it compliant with the DISDRODB standards.
  All reader scripts live in the `disdrodb/l0/readers <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers>`_ directory,
  organized by sensor name and data source: ``disdrodb/l0/readers/<sensor_name>/<DATA_SOURCE>/<READER_NAME>.py``.
  To point the disdrodb software to the correct reader, the ``reader`` reference must be defined as ``<DATA_SOURCE>/<READER_NAME>``.

  For example, to select the OTT Parsivel GPM IFLOODS reader (defined at
  `disdrodb.l0.readers.PARSIVEL.GPM.IFLOODS.py <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers/PARSIVEL/GPM/IFLOODS.py>`_)
  the ``reader`` reference ``GPM/IFLOODS`` must be used.


The ``disdrodb_data_url`` metadata key references to the remote/online repository where station's raw data are stored.
At this URL, a single zip file provides all data available for a given station.

To check the validity of the metadata YAML files, run the following code:

.. code-block:: python

    from disdrodb import check_metadata_archive, check_metadata_archive_geolocation

    check_metadata_archive()
    check_metadata_archive_geolocation()


The list and description of the DISDRODB metadata is provided here below:


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
