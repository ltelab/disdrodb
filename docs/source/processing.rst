.. _processing:

============================
Archive Processing
============================

DISDROB enables to process the data from the command line or by calling a python function.
The following sections describe how to use the disdrodb software to generate DISDRODB L0B netCDF files.


Launch DISDRODB L0 processing for a specific station
======================================================


**Command line solution**


.. code-block:: bash

	disdrodb_run_l0_station <data_source> <campaign_name> <station_name> [parameters]


Example:

.. code-block:: bash

	disdrodb_run_l0_station EPFL_2008 10 --l0a_processing True --l0b_processing True --force True --verbose True --parallel False

Type ``disdrodb_run_l0_station --help`` in the terminal to get more information on the possible parameters.


**Pythonic solution**


.. code-block::

    from disdrodb.l0 import run_disdrodb_l0_station

    run_disdrodb_l0_station(data_source, campaign_name, station_name, **kwargs)


Example :


.. code-block:: python

    from disdrodb.l0 import run_disdrodb_l0_station
    from disdrodb.configs import get_data_archive_dir

    data_archive_dir = get_data_archive_dir()
    data_source = "EPFL"
    campaign_name = "EPFL_2008"
    station_name = "10"

    # L0 processing settings
    l0a_processing = True
    l0b_processing = True
    l0c_processing = True
    remove_l0a = False
    remove_l0b = False

    # L0 processing options
    force = True
    verbose = True
    debugging_mode = True
    parallel = False

    # Run the processing
    run_disdrodb_l0_station(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # L0 processing settings
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        l0c_processing=l0c_processing,
        remove_l0a=remove_l0a,
        remove_l0b=remove_l0b,
        # L0 processing options
        parallel=parallel,
        verbose=verbose,
        force=force,
        debugging_mode=debugging_mode,
    )


Launch DISDRODB L0 processing for a set of stations
==================================================================


DISDRODB offers an utility to run the process of multiple stations with a single command.

In the code example belows, if ``--data_sources``, ``--campaign_names``, ``--station_names``
are not specified, the command will process all stations available within the local DISDRODB Data Archive.
Starting from all the available stations, the optional specification of the ``--data_sources`` , ``--campaign_names``
and ``--station_names`` will restrict the stations that will be processed.

For example:

- if only the ``--campaign_names`` argument is specified, DISDRODB will process only the stations of such campaigns.
- if only the ``--data_sources`` argument is specified, DISDRODB will process all the stations of such data sources.
- if only the ``--station_names`` argument is specified, DISDRODB will process only the specified stations.


**Command line solution**



.. code-block:: bash

	disdrodb_run_l0 --data_sources <data_sources> --campaign_names <campaign_names> --station_names <station_names> [parameters]

Example :

.. code-block:: bash

	disdrodb_run_l0 --campaign_names EPFL_2008 --l0a_processing True --l0b_processing True --parallel False

To  specify multiple campaigns you can do the follow

.. code-block:: bash

	disdrodb_run_l0  --campaign_names 'EPFL_2008 LOCARNO_2018' --l0a_processing True --l0b_processing True --parallel False


Type ``disdrodb_run_l0 --help`` in the terminal to get more information on the possible parameters.


**Pythonic solution**


.. code-block::

    from disdrodb.l0 import run_disdrodb_l0

    run_disdrodb_l0(data_source, campaign_name, **kwargs)


Example :

.. code-block:: python

    from disdrodb.l0 import run_disdrodb_l0
    from disdrodb.configs import get_data_archive_dir

    data_archive_dir = get_data_archive_dir()
    data_sources = ["EPFL"]
    campaign_names = ["EPFL_2008"]

    # L0 processing settings
    l0a_processing = True
    l0b_processing = True
    l0c_processing = True
    remove_l0a = False
    remove_l0b = False
    # L0 processing options
    force = True
    verbose = True
    debugging_mode = True
    parallel = False

    run_disdrodb_l0(
        data_archive_dir=data_archive_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        # station_names=station_names,
        # L0 processing settings
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        l0c_processing=l0c_processing,
        remove_l0a=remove_l0a,
        remove_l0b=remove_l0b,
        # L0 processing options
        parallel=parallel,
        verbose=verbose,
        force=force,
        debugging_mode=debugging_mode,
    )


.. note::

    This documentation section is in preparation and will be updated soon !

Other arguments to describe

* ``station_name`` : str - Name of the station to be processed.


* ``force`` : bool [true\| **false** ] - Whether to overwrite existing data.

    *  If ``True``, overwrite existing data into destination directories.
    *  If ``False``, raise an error if there are already data into destination directories.


* ``verbose`` : bool [true\| **false** ] - Whether to print detailed processing information into terminal.


* ``debugging_mode`` : bool [true\| **false** ] -  If ``True``, it reduces the amount of data to process.

    * It processes just 3 raw data files.

* ``parallel`` : bool [true\| **false** ] - Whether to process multiple files simultaneously.

    * If ``parallel=False``, the raw files are processed sequentially.
    * If ``parallel=True``, each file is processed in a separate core.
