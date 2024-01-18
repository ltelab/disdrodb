=========================
DISDRODB Data Download
=========================

In this section, we describe how to download disdrometer data from the DISDRODB Decentralized Data Archive to your local machine.
First, is however necessary to download on your local machine the DISDRODB Metadata Archive, which contains the pointers
to the remote data repositiores where the DISDRODB stations are stored.

.. note:: The DISDRODB Metadata Archive is often updated with new stations or metadata.
          Therefore, we recommend to regularly update your local DISDRODB Metadata Archive (see below).

Download the official DISDRODB Metadata Archive
-----------------------------------------------

First travel to the directory where you want to store the DISDRODB Data Archive with :code:`cd <the_root_directory>`

Then clone the DISDRODB Metadata Archive repository with:

.. code:: bash

   git clone https://github.com/ltelab/disdrodb-data.git

This will create a directory called ``disdrodb-data``.

.. note:: Remember that the DISDRODB Metadata Archive is often updated with new stations or metadata.
          To update your local DISDRODB Metada Archive (and therefore download recently added new stations),
          run :code:`git pull` inside the ``disdrodb-data`` directory.


Define the DISDRODB Base Directory
------------------------------------------

The DISDRODB base directory is the directory ``DISDRODB`` inside ``disdrodb-data``.

You can set the default DISDRODB base directory by running in python:

.. code:: python

    import disdrodb

    base_dir = "<path_to>/disdrodb-data/DISDRODB>"
    disdrodb.define_configs(base_dir=base_dir)

By running this command, the disdrodb software will write a ``.config_disdrodb.yml`` file into your home directory (i.e. ``~/.config_disdrodb.yml``)
that will be used as default configuration file when running the disdrodb software.


Alternatively, you can also define the DISDRODB base directory as an environment variable ``DISDRODB_BASE_DIR``.
In the terminal, you must type the following command:

.. code:: bash

   export DISDRODB_BASE_DIR="<path_to>/disdrodb-data/DISDRODB"

.. note:: It's important to remember that the environment variable ``DISDRODB_BASE_DIR`` (if defined) will take priority over the default path
          defined in the ``.config_disdrodb.yml`` file.


Download the DISDRODB Data Archive
---------------------------------------

In order to download the data, you should be in a virtual environment with the disdrodb package installed!
Refers to the installation section for more details on how to set-up and activate the virtual environment.

To download all data stored into the DISDRODB Decentralized Data Archive, you just have to run the following command:

.. code:: bash

   disdrodb_download_archive  --data_sources <data_source> --campaign_names <campaign_name> --station_names <station_name> --force true

The ``data_sources``, ``campaign_names`` and ``station_names`` parameters are optional and are meant to restrict the download to a specific set of
data sources, campaigns, and/or stations.

Parameters:

-  ``data_sources`` (optional): Station data sources.
-  ``campaign_names`` (optional): Station campaign names.
-  ``station_names`` (optional): Name of the stations.
-  ``force`` (optional, default = ``False``): a boolean value indicating
   whether existing files should be overwritten.

To download data from multiple data sources, campaigns, or stations, please provide a space-separated string of
the data sources, campaigns or stations you require.

For example:

* if you want to download all EPFL and NASA data use ``--data_sources "EPFL NASA"``,

* if you want to download stations of specific campaigns, use ``--campaign_names "HYMEX_LTE_SOP3 HYMEX_LTE_SOP4"``.

* if you want to download stations named in a specific way, use ``--station_names "station1 station2"``.
