.. _metadata_archive:

==========================
Metadata Archive
==========================

**What is the DISDRODB Metadata Archive?**

The `DISDRODB Metadata Repository <https://github.com/ltelab/DISDRODB-METADATA>`__ is a centralized,
GitHub-hosted repository that serves as the hub for:

- Tracking all available disdrometer stations worldwide
- Documenting potential sensor malfunctions and data quality issues
- Listing URLs of remote repositories storing raw disdrometer data
- Maintaining comprehensive metadata for each station

**Benefits of Community-Driven Metadata**

This platform facilitates community collaboration to continuously enhance station metadata
using best open-source practices.
It enables iterative data quality improvements while keeping the DISDRODB
processing chain transparent and fully reproducible.

**Metadata Standards and Quality Control**

To ensure metadata quality and consistency, a comprehensive set of standardized metadata keys
has been established.
The DISDRODB community can identify timestamps or periods when sensors malfunctioned
or produced erroneous data using dedicated YAML issue files.

**Regular Updates**

The DISDRODB Metadata Repository is regularly updated to reflect:

- Current station status (active, inactive, decommissioned)
- Data availability and coverage periods
- Known issues and quality flags

**Contributing to the Metadata Archive**

The sections below detail the steps required to add or update information in the DISDRODB Metadata Archive.


Download the Metadata Archive
----------------------------------

**When to clone the Metadata Archive:**

- You want to add new data to the DISDRODB Decentralized Data Archive
- You need to update station metadata
- You want to add or modify issue information

**Steps:**

1. Fork the repository on GitHub (click "Fork" button on the repository page)
2. Clone your fork to your local machine:

.. code:: bash

   git clone https://github.com/<your_username>/DISDRODB-METADATA.git


Update the Metadata Archive
-------------------------------

**Step-by-Step Guide**

Follow these steps to contribute updates to the DISDRODB Metadata Archive:

**1. Navigate to the Repository**

Change to the ``DISDRODB-METADATA`` directory where you cloned the repository:

.. code:: bash

   cd DISDRODB-METADATA

**2. Create a New Branch**

Create a descriptive branch name based on your contribution:

.. code:: bash

   git checkout -b "add-metadata-<data_source>-<campaign_name>"

.. note::
   **Branch Naming Conventions:**

   - Adding a new station: ``add-metadata-<data_source>-<campaign_name>``
   - Updating existing station: ``update-metadata-<data_source>-<campaign_name>-<station_name>``

**3. Edit or Add Metadata Files**

Make your changes to the metadata files. You can use these commands to navigate to the relevant directories:

**Open the Metadata Archive:**

.. code:: bash

   disdrodb_open_metadata_archive

**Open a specific station directory:**

.. code:: bash

   disdrodb_open_metadata_directory <DATA_SOURCE> <CAMPAIGN_NAME> <STATION_NAME>

**4. Verify Your Changes**

Before committing, validate that your metadata files comply with DISDRODB standards:

.. code:: bash

   export DISDRODB_METADATA_ARCHIVE_DIR="<path_to>/DISDRODB-METADATA/DISDRODB"
   disdrodb_check_metadata_archive

.. note::
   You only need to set the ``DISDRODB_METADATA_ARCHIVE_DIR`` environment variable if you haven't
   already configured the archive directory in your DISDRODB configuration file.

**5. Commit and Push Your Changes**

Commit your changes with a descriptive message and push to your fork:

.. code:: bash

   git add .
   git commit -m "Add/update metadata for <data_source> <campaign_name>"
   git push origin <branch_name>

.. note::
   Replace ``<data_source>``, ``<campaign_name>``, and ``<branch_name>`` with your actual values.

**6. Create a Pull Request**

Open a pull request on the GitHub DISDRODB Metadata Repository:

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Provide a clear description of your changes
4. Submit the pull request and wait for a maintainer to review it

For detailed instructions, see the `Creating a Pull Request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__ documentation.

**7. Get Help if Needed**

If you encounter any issues during this process, please `open an issue <https://github.com/ltelab/DISDRODB-METADATA/issues/new/choose>`_ and we'll be happy to assist!


Validate the Metadata Archive
--------------------------------

**Automated Quality Checks**

You can verify that all station metadata comply with DISDRODB standards using the following methods:

**Using Python:**

.. code:: python

    import disdrodb

    disdrodb.check_metadata_archive()

**Using the Terminal:**

.. code:: bash

   disdrodb_check_metadata_archive


Explore the Metadata Archive
--------------------------------

**Load Metadata into a DataFrame**

The disdrodb software provides the ``read_metadata_archive()`` function to load the
entire metadata archive into a ``pandas.DataFrame`` for easy exploration and analysis:

.. code:: python

    import disdrodb

    df = disdrodb.read_metadata_archive()
    print(df)
