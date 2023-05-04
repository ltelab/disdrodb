=========================
Data
=========================


Users can make their own data accessible to the community. DISDRODB
provides a central storage for code (readers), issues and metadata.
However, the raw data itself must be stored by the data provider due to
size limitations.

Two types of data must be distinguished:

-  Station Raw Data:

   -  Stores disdrometer measurements for days, weeks, and years.
   -  This dataset can be very heavy.
   -  No central storage is provided.

-  Station Metadata and Issues:

   -  Stores a standard set of metadata and measurement issues of each disdrometer 
   -  This dataset should be light.
   -  Central storage is provided in the ``disdro-data`` Git repository.
   -  The metadata folder contains a YAML metadata file called
      ``metadata.yml``. It has a ``data_url`` key that references to the remote/online repository where
      station's raw data are stored. At this URL, a single zip file provides all data available for a given station.

How to download metadatas and issues ?
--------------------------------------

You can clone the disdrodb-data repository with

.. code:: bash

   git clone https://github.com/ltelab/disdrodb-data.git

However, if you plan to add new data or metadata to the archive, first
fork the repository on your GitHub account and then clone the forked
repository.

How to upload metadatas and issues?
------------------------------------

Do you want to contribute to the project with your own data? Great! Just
follow these steps:

1. Fork the ``disdro-data`` Git repository.

2. Create a new branch.

   .. code:: bash

      git checkout -b "reader-<data_source>-<campaign_name>"

3. Add your data source, campaign names, and station name to the current
   folder structure.

4. Add your metadata YAML file for each station ``station_name.yml``, in
   a metadata directory in the campaign directory. We recommend you
   copy-paste an existing metadata YAML file to get the correct
   structure.

5. (Optional) Add your issues YAML files, for each station
   ``station_name.yml``, in an issue directory located in the campaign
   directory. We recommend you copy-paste an existing metadata YAML file
   to get the correct structure.

6. Test that the integration of your new dataset functions by deleting
   your data locally and re-fetching it through the process detailed
   above.

7. Commit your changes and push your branch to GitHub.

8. `Create a pull
   request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__,
   and wait for a maintainer to accept it!

9. If you struggle with this process, don’t hesitate to raise an
   `issue <https://github.com/ltelab/disdrodb-data/issues/new/choose>`__
   so we can help!

How to download raw data locally ?
--------------------------------------

Prerequisite: You have already set up the folder hierarchy as described
above.

Objective: You would like to download the raw data referenced in the
``metadata.yml`` file.

In order to download data, you should be in the virtual environment.

To download all data, just run:

.. code:: bash

   download_disdrodb_archive  <the_root_folder> --data_sources <data_souzrce> --campaign_names <capaign_name> --station_names <station_name> --force true

The ``disdrodb_dir`` parameter is compulsory and must include the path
of the root folder, ending with ``DISDROD``. The other parameters are
optional and are meant to restrict the download processing to a specific
data source, campaign, or station.

Parameters:

-  ``data_sources`` (optional): the source of the data
-  ``campaign_names`` (optional): the name of the campaign.
-  ``station_names`` (optional): the name of the station.
-  ``force`` (optional, default = ``False``): a boolean value indicating
   whether existing files should be overwritten.

How to upload raw data to Zenodo?
---------------------------------

We provide users with a code to upload their station’s raw data to
Zenodo.

.. code:: bash

   upload_disdrodb_archive <the_root_folder> --data_sources <data_source> --campaign_names <campaign_name> --station_names <station_name> --platform <name_of_the_platform> --force true

The ``disdrodb_dir`` parameter is compulsory and must include the path
of the root folder, ending with ``DISDROD``. The other parameters are
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

Currently, only Zenodo is supported.

After running this command, the user will be prompted to insert a Zenodo
token. Once the data is uploaded, a link will be displayed that the user
must use to go to the Zenodo web interface and manually publish the
data.

To get a Zenodo token, go to
https://zenodo.org/account/settings/applications/tokens/new/

.. image:: /static/zenodo.png
