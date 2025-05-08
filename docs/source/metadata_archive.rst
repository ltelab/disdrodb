.. _metadata_archive:

==========================
Metadata Archive
==========================

The `DISDRODB metadata repository <https://github.com/ltelab/DISDRODB-METADATA>`__ is hosted on GitHub and serves as a central hub for tracking available stations,
the potential malfunctioning of the sensors, and to list the URLs of the remote data repositories where the raw disdrometer data are stored.

The GitHub platform facilitates community collaboration to continuously enhance station metadata using best open-source practices.
This approach also enables recursive data quality improvements, while keeping the DISDRODB product chain transparent and fully reproducible.

To ensure quality and consistency of metadata, a comprehensive standard set of metadata keys has been established.
The DISDRODB community is empowered to pinpoint specific timestamps or periods when sensors might have malfunctioned or generated erroneous data logs through specific issues YAML files.

The DISDRODB Metadata Repository is therefore updated on a regular basis to reflect the latest status of the stations and the data availability.

Here below we detail the necessary step to add/update the information of the DISDRODB Metadata Archive.


Download the Metadata Archive
----------------------------------

If you plan to add new data to the DISDRODB Decentralized Data Archive or you want to just update
some station metadata/issues information, go to the
`DISDRODB metadata repository <https://github.com/ltelab/DISDRODB-METADATA>`__,
fork the repository on your GitHub account and then clone the forked repository:

.. code:: bash

   git clone https://github.com/<your_username>/DISDRODB-METADATA.git


Update the Metadata Archive
-------------------------------

To update the DISDRODB Metadata Archive follow these steps:

1. Go inside the ``DISDRODB-METADATA`` directory where you have cloned the repository:

2. Create a new branch.

   .. code:: bash

      git checkout -b "add-metadata-<data_source>-<campaign_name>"

   .. note::
      If you are adding information regarding a new station, please name the branch as follows: ``add-metadata-<data_source>-<campaign_name>``.

      If you are just improving some specific information of an existing station, please name the branch as follows: ``update-metadata-<data_source>-<campaign_name>-<station_name>``.

3. Edit or add the metadata files that you are interested in.

   To quickly open the metadata data archive, you can type in the terminal:

   .. code:: bash

      disdrodb_open_metadata_archive

   To quickly open the metadata directory of a specific station, you can type in the terminal:

   .. code:: bash

      disdrodb_open_metadata_directory <DATA_SOURCE> <CAMPAIGN_NAME> <STATION_NAME>


4. When you are done, please run the following command to check that the metadata files are valid:

   .. code:: bash

      export DISDRODB_METADATA_ARCHIVE_DIR="<path_to>/DISDRODB-METADATA/DISDRODB"
      disdrodb_check_metadata_archive

   .. note::
      The ``DISDRODB_METADATA_ARCHIVE_DIR`` environment variable has to be specified only if the DISDRODB Metadata Archive directory has not been specified before in the DISDRODB Configuration File.

5. Commit your changes and push your branch to GitHub:

   .. code:: bash

      git add *
      git commit -m "Add/update metadata for <data_source> <campaign_name>"
      git push origin <branch_name>

6. Go to the `GitHub DISDRODB Metadata Repository <https://github.com/ltelab/DISDRODB-METADATA>`__, open the Pull Request and wait for a maintainer to accept it!
   For more information on GitHub Pull Requests, read the
   `"Create a pull request documentation" <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__.

7.  If you struggle with this process, do not hesitate to raise an `issue <https://github.com/ltelab/DISDRODB-METADATA/issues/new/choose>`__ so we can help!


Check the Metadata Archive
--------------------------------

You can check that all stations metadata adhere to the enforced DISDRODB standards by running the following command in python:

.. code:: python

    import disdrodb

    check_metadata_archive()


Alternatively you can type into the terminal:

.. code:: bash

   disdrodb_check_metadata_archive


Explore the Metadata Archive
--------------------------------

The disdrodb software provides the ``read_metadata_archive`` function to read entire metadata archive
into a ``pandas.DataFrame``:


.. code:: python

    import disdrodb

    df = disdrodb.read_metadata_archive()
    print(df)
