==============================
How to Contribute New Data
==============================

Users can make their own data accessible to the community.
DISDRODB provides a central storage for code (readers), issues and metadata.
However, the raw data itself must be stored by the data provider on a remote data
repository (e.g., Zenodo, Figshare, etc.).


Two types of data must be distinguished:

-  DISDRODB Raw Data:

   -  Contain disdrometer measurements for days, weeks, and years.
   -  This data can be very large. No central storage is provided.
   -  DISDRODB provides utility functions to easily upload the raw data on remote data
      repositories (i.e. Zenodo)
   -  DISDRODB provides utility functions to download the raw data from the remote data repositories.

-  DISDRODB Metadata and Issues YAML files:

   -  Each disdrometer station has a standardized metadata and issue YAML file.
   -  The ``disdrodb_data_url`` metadata key references to the remote/online repository where
   -  station's raw data are stored. At this URL, a single zip file provides all data available for a given station.
   -  The DISDRODB Metadata Archive, hosted on the ``disdro-data`` GitHub repository, acts as a centralized storage
      for the metadata and issue YAML files of all DISDRODB stations.


Data transfer upload and download schema:

.. image:: /static/transfer.png


Add stations information to the DISDRODB metadata archive
----------------------------------------------------------

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

8. Go to the `Github DISDRODB Metadata Repository <https://github.com/ltelab/disdrodb-data>`__, open the Pull Request and wait for a maintainer to accept it!
   For more information on GitHub Pull Requests, read the
   `"Create a pull request documentation" <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__.

9.  If you struggle with this process, do not hesitate to raise an `issue <https://github.com/ltelab/disdrodb-data/issues/new/choose>`__ so we can help!



Upload your stations data on Zenodo and link it to the DISDRODB Decentralized Data Archive
----------------------------------------------------------------------------------------------

We provide users with a code to easily upload their stations raw data to `Zenodo <https://zenodo.org/>`_.

.. code:: bash

   upload_disdrodb_archive <the_root_folder> --data_sources <data_source> --campaign_names <campaign_name> --station_names <station_name> --platform <name_of_the_platform> --force true

The ``base_dir`` parameter is compulsory and must include the path
of the root folder, ending with ``DISDRODB``. The other parameters are
optional and are meant to restrict the upload processing to a specific
data source, campaign, or station.

Parameters:

-  ``data_sources`` (optional): the source of the data.
-  ``campaign_names`` (optional): the name of the campaign.
-  ``station_names`` (optional): the name of the station.
-  ``platform`` (optional, default is Zenodo).
    Currently, only Zenodo is supported.
-  ``force`` (optional, default = ``False``): a boolean value indicating
   whether files already uploaded somewhere else should still be
   included.

To upload data from multiple data sources or campaigns, please provide a space-separated string of
the data sources or campaigns you require. For example, ``"EPFL NASA"``.

After running this command, the user will be prompted to insert a Zenodo
token. Once the data is uploaded, a link will be displayed that the user
must use to go to the Zenodo web interface and manually publish the
data.

To get a Zenodo token, go to
`https://zenodo.org/account/settings/applications/tokens/new/ <https://zenodo.org/account/settings/applications/tokens/new/>`_




.. image:: /static/zenodo.png



Test the download and DISDRODB L0 processing of the stations you contributed
------------------------------------------------------------------------------

To test that the data upload has been successfuland you specified correctly all the required metadata, let's first try to download
the data you just uploaded from the DISDRODB Decentralized Data Archive.

To do so, first make a copy of the DISDRODB metadata archive you just edited, in order to inadvertently delete the data you just uploaded.

Then, run the following command to download the data you just uploaded:

.. code:: 

   export DISDRODB_BASE_DIR="<the_path_to_a_copy_of_the_disdrodb-data_you_edited>/DISDRODB"
   disdrodb_download_archive  --data_sources <your_data_source> --campaign_names <your_new_campaign> --force true

::note
   Be sure to specify a ``DISDRODB_BASE_DIR`` environment variable that points to a copy of the metadata archive you edited
   otherwise you risk to overwrite the data you just uploaded!

If the download is successful, and you also already implemented the DISDRODB reader for your data, you can now try to process the data you just downloaded.

To do so, run the following command:

.. code:: 

   export DISDRODB_BASE_DIR="<the_path_to_a_copy_of_the_disdrodb-data_you_edited>/DISDRODB"
   disdrodb_run_l0  --data_sources <your_data_source> --campaign_names <your_new_campaign>

   ::note
      If the correctness of the reader has already been tested, you can add the ``--debugging_mode True`` parameter to just run the processing
      on a small subset of the data.  This will speed up the processing and will allow you to check that the processing is working correctly.


If the processing is successful, you can now open a Pull Request to merge your changes to the DISDRODB metadata archive.
Congratulations !!! Your data are now available to the community !!!
