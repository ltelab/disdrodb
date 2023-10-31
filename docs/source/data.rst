=====
Data
=====


Users can make their own data accessible to the community. DISDRODB
provides a central storage for code (readers), issues and metadata.
However, the raw data itself must be stored by the data provider due to
size limitations.

Two types of data must be distinguished:

-  Station Raw Data:

   -  Stores disdrometer measurements for days, weeks, and years.
   -  This dataset can be very heavy.
   -  No central storage is provided.

-  Station Metadata and Issues YAML files:

   -  Stores a standard set of metadata and measurement issues of each disdrometer.
   -  Central storage is provided in the ``disdro-data`` Git repository.
   -  The ``/metadata`` folder contains a YAML metadata file called
      ``<station_name>.yml``. It has a ``data_url`` key that references to the remote/online repository where station's raw data are stored. At this URL, a single zip file provides all data available for a given station.


Data transfer upload and download schema:

.. image:: /static/transfer.png


Download the DISDRODB metadata archive
-----------------------------------------

First travel to the directory where you want to store the data.
Then clone the disdrodb-data repository with:

.. code:: bash

   git clone https://github.com/ltelab/disdrodb-data.git

However, if you plan to add new data or metadata to the archive, first
fork the repository on your GitHub account and then clone the forked
repository.

Update the DISDRODB metadata archive
----------------------------------------

Do you want to contribute to the project with your own data? Great! Just
follow these steps:

1. Fork the ``disdro-data`` Git repository.

2. Create a new branch:

   .. code:: bash

      git checkout -b "reader-<data_source>-<campaign_name>"

3. Add your data source and campaign name directory to the current
   disdrodb-data structure.

4. Add your metadata YAML file for each station (following the name format convention ``<station_name>.yml``) in the ``metadata`` directory of the campaign directory. We recommend you to copy-paste an existing metadata YAML file to get the correct structure.

5. (Optional) Add your issues YAML files, for each station
   ``station_name.yml``, in an ``issues`` directory located in the campaign
   directory. We recommend you to copy-paste an existing issue YAML file
   to get the correct structure.

6. Commit your changes and push your branch to GitHub.

7. Test that the integration of your new dataset functions by deleting
   your data locally and re-fetching it through the process detailed above.

8. `Create a pull
   request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__,
   and wait for a maintainer to accept it!

9.  If you struggle with this process, don’t hesitate to raise an `issue <https://github.com/ltelab/disdrodb-data/issues/new/choose>`__ so we can help!

Download the DISDRODB raw data archive
---------------------------------------

Prerequisite: First clone the disdrodb-data repository as described above to get the DISDRODB directory structure.
Objective: You would like to download the raw data referenced in some metadata ``<station_name>.yml`` file.

In order to download the data, you should be in a virtual environment with the disdrodb package installed!

To download all data, just run:

.. code:: bash

   download_disdrodb_archive  <the_root_folder> --data_sources <data_source> --campaign_names <campaign_name> --station_names <station_name> --force true

The ``disdrodb_dir`` parameter is compulsory and must include the path
of the root folder, ending with ``DISDRODB``. The other parameters are
optional and are meant to restrict the download processing to a specific
data source, campaign, or station.

Parameters:

-  ``data_sources`` (optional): Station data source.
-  ``campaign_names`` (optional): Station campaign name.
-  ``station_names`` (optional): Name of the stations.
-  ``force`` (optional, default = ``False``): a boolean value indicating
   whether existing files should be overwritten.

To download data from multiple data sources or campaigns, please provide a space-separated string of
the data sources or campaigns you require. For example, ``"EPFL NASA"``.


Add new stations raw data to the DISDRODB archive (using Zenodo)
-----------------------------------------------------------------

We provide users with a code to upload their station’s raw data to `Zenodo <https://zenodo.org/>`_.

.. code:: bash

   upload_disdrodb_archive <the_root_folder> --data_sources <data_source> --campaign_names <campaign_name> --station_names <station_name> --platform <name_of_the_platform> --force true

The ``disdrodb_dir`` parameter is compulsory and must include the path
of the root folder, ending with ``DISDRODB``. The other parameters are
optional and are meant to restrict the upload processing to a specific
data source, campaign, or station.

Parameters:

-  ``data_sources`` (optional): the source of the data.
-  ``campaign_names`` (optional): the name of the campaign.
-  ``station_names`` (optional): the name of the station.
-  ``platform`` (optional, default is Zenodo).
-  ``force`` (optional, default = ``False``): a boolean value indicating
   whether files already uploaded somewhere else should still be
   included.

To upload data from multiple data sources or campaigns, please provide a space-separated string of
the data sources or campaigns you require. For example, ``"EPFL NASA"``.


Currently, only Zenodo is supported.

After running this command, the user will be prompted to insert a Zenodo
token. Once the data is uploaded, a link will be displayed that the user
must use to go to the Zenodo web interface and manually publish the
data.

To get a Zenodo token, go to
`https://zenodo.org/account/settings/applications/tokens/new/ <https://zenodo.org/account/settings/applications/tokens/new/>`_




.. image:: /static/zenodo.png
