.. _metadata:

=========================
Station Metadata
=========================

The metadata for each station are defined in a YAML file that uses a standardized set of keys.

It is mandatory to at least specify values for the following 7 metadata keys:

* ``data_source``: must match the data source where the metadata reside.
* ``campaign_name``: must match the campaign name where the metadata reside.
* ``station_name``: must match the YAML filename (excluding the ``.yml`` extension).
* ``sensor_name``: must be one of the configured sensors (see ``disdrodb.available_sensor_names()``). If your sensor is not listed, follow :ref:`Add new sensor configs <sensor_configurations>`.
* ``reader``: indicates which function ingests the raw data. Readers live in ``disdrodb/l0/readers/<sensor_name>/<DATA_SOURCE>/<READER_NAME>.py``. Set ``reader`` to ``<DATA_SOURCE>/<READER_NAME>`` (e.g. ``NASA/IFLOODS`` for the OTT Parsivel NASA IFLOODS reader).
* ``raw_data_format``: choose ``txt`` for text/ASCII files or ``netcdf`` for netCDF files.
* ``raw_data_glob_pattern``: a glob pattern that selects which files in ``DISDRODB/RAW/<DATA_SOURCE>/<CAMPAIGN_NAME>/<STATION_NAME>/data`` are ingested. For example, ``*.txt`` matches all ``.txt`` files recursively. To match only files with a specific prefix, use ``SPECTRUM_*.txt``. To limit to a subfolder, include its name: ``custom/*.txt`` (direct files only) or ``custom/**/*.txt`` (including nested folders).
* ``measurement_interval``: the sensor measurement sampling interval in seconds.
* ``deployment_status``: either ``'ongoing'`` or ``'terminated'``.
* ``deployment_mode``: possible values are ``'land'``, ``'ship'``, ``'truck'`` or ``'cable'``.
* ``platform_type``: choose ``fixed`` or ``mobile``. Use ``mobile`` if the platform's latitude, longitude, or altitude changes over time.

The ``disdrodb_data_url`` metadata key specifies the URL of the remote repository where raw data are stored. This link should point to a zip file containing all data for the station.

To check the validity of the metadata YAML files, run the following code:

.. code-block:: python

    from disdrodb import check_metadata_archive, check_metadata_archive_geolocation

    check_metadata_archive()
    check_metadata_archive_geolocation()


Below is the list and description of DISDRODB metadata keys:


.. csv-table:: Mandatory keys
   :align: left
   :file: ./metadata_csv/Mandatory_Keys.csv
   :widths: auto
   :header-rows: 1


.. csv-table:: Deployment info
   :align: left
   :file: ./metadata_csv/Deployment_Info.csv
   :widths: auto
   :header-rows: 1


.. csv-table:: Station description
   :align: left
   :file: ./metadata_csv/Description.csv
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
