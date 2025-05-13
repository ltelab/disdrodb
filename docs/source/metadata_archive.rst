.. _metadata_archive:

==========================
Metadata Archive
==========================

The `DISDRODB metadata repository <https://github.com/ltelab/DISDRODB-METADATA>`__ is hosted
on GitHub and serves as a central hub for tracking available stations, potential sensor malfunctions,
and listing the URLs of remote repositories storing raw disdrometer data.

This platform facilitates community collaboration to continuously enhance station metadata
using best open-source practices.
It also enables iterative data quality improvements while keeping the DISDRODB
 product chain transparent and fully reproducible.

To ensure the quality and consistency of metadata, a comprehensive standard set of metadata keys
has been established.
The DISDRODB community can pinpoint timestamps or periods when sensors may have malfunctioned
or produced erroneous data using dedicated YAML issue files.

Consequently, the DISDRODB Metadata Repository is regularly updated to reflect station status
and data availability.

Below we detail the steps required to add or update information in the DISDRODB Metadata Archive.


Download the Metadata Archive
----------------------------------

If you plan to add new data to the DISDRODB Decentralized Data Archive or simply want
to update station metadata or issue information, fork the repository on GitHub and then clone your fork:

.. code:: bash

   git clone https://github.com/<your_username>/DISDRODB-METADATA.git


Update the Metadata Archive
-------------------------------

Follow these steps to update the DISDRODB Metadata Archive:

1. Navigate to the ``DISDRODB-METADATA`` directory where you cloned the repository.

2. Create a new branch.

   .. code:: bash

      git checkout -b "add-metadata-<data_source>-<campaign_name>"

   .. note::
      If you are adding information about a new station, name the branch ``add-metadata-<data_source>-<campaign_name>``.
      If you are improving data for an existing station, name it ``update-metadata-<data_source>-<campaign_name>-<station_name>``.

3. Edit or add the desired metadata files.

   To open the DISDRODB Metadata Archive directory, type in the terminal:

   .. code:: bash

      disdrodb_open_metadata_archive

   To open a specific station directory:

   .. code:: bash

      disdrodb_open_metadata_directory <DATA_SOURCE> <CAMPAIGN_NAME> <STATION_NAME>

4. When done, run this command to verify the metadata files:

   .. code:: bash

      export DISDRODB_METADATA_ARCHIVE_DIR="<path_to>/DISDRODB-METADATA/DISDRODB"
      disdrodb_check_metadata_archive

   .. note::
      You only need to set ``DISDRODB_METADATA_ARCHIVE_DIR`` if you have not already specified the archive directory in your DISDRODB configuration.

5. Commit and push your changes to GitHub:

   .. code:: bash

      git add *
      git commit -m "Add/update metadata for <data_source> <campaign_name>"
      git push origin <branch_name>

6. Open a pull request on the GitHub DISDRODB Metadata Repository and wait for a maintainer to review and merge it.
   For more details, see the `"Creating a pull request" <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__ documentation.

7.  If you encounter any issues, feel free to `raise one <https://github.com/ltelab/DISDRODB-METADATA/issues/new/choose>`_ so we can assist!


Check the Metadata Archive
--------------------------------

You can verify that all station metadata adhere to DISDRODB standards by running the following command in Python:

.. code:: python

    import disdrodb

    check_metadata_archive()


Alternatively, run the following command in the terminal:

.. code:: bash

   disdrodb_check_metadata_archive


Explore the Metadata Archive
--------------------------------

The disdrodb software provides the ``read_metadata_archive`` function to load the
entire metadata archive into a ``pandas.DataFrame``:

.. code:: python

    import disdrodb

    df = disdrodb.read_metadata_archive()
    print(df)
