============================
Run DISDRODB L0 processing
============================

Launch DISDRODB L0 processing for a specific station
======================================================


There are two ways of process a station using DISDRODB.


1. By command line :


	.. code-block::

		run_disdrodb_l0_station <disdrodb_dir> <data_source> <campaign_name> <station_name> [parameters]


	Example :

	.. code-block::

		run_disdrodb_l0_station /ltenas8/DISDRODB EPFL  EPFL_2008 10 --l0a_processing True --l0b_processing True --force True --verbose True --parallel False

    Type ``run_disdrodb_l0_station --help`` in the terminal to get more information on the possible parameters.


2. By calling a python function


	.. code-block:: python

		from disdrodb.l0 import run_disdrodb_l0_station
		run_disdrodb_l0_station(<disdrodb_dir> <data_source>, <campaign_name>, <station_name>, ...)


	Example :

	.. code-block:: python

		from disdrodb.l0 import run_disdrodb_l0_station

		disdrodb_dir = "...\\DISDRODB"
		data_source='EPFL'
		campaign_name='EPFL_2008'
		station_name="10"

		# L0 processing settings
		l0a_processing=True
		l0b_processing=True
		l0b_concat=True
		remove_l0a=False
		remove_l0b=False

		# L0 processing options
		force=True
		verbose=True
		debugging_mode=True
		parallel=False
		# Run the processing

		run_disdrodb_l0_station(
			disdrodb_dir=disdrodb_dir,
			data_source=data_source,
			campaign_name=campaign_name,
			station_name=station_name,
			# L0 processing settings
			l0a_processing=l0a_processing,
			l0b_processing=l0b_processing,
			l0b_concat=l0b_concat,
			remove_l0a=remove_l0a,
			remove_l0b=remove_l0b,
			# L0 processing options
			parallel=parallel,
			verbose=verbose,
			force=force,
			debugging_mode=debugging_mode,
		)


Launch DISDRODB L0 processing for all stations within a campaign
==================================================================


DISDRODB offers an utility to run the process of multiple stations with a single command.

In the code example belows, if ``--data_sources``, ``--campaign_names``, ``--station_names``
are not specified, the command will process all stations available within the ``<disdrodb_dir>``.
Starting from all the available stations, the optional specification of the ``--data_sources`` , ``--campaign_names``
and ``--station_names`` will restrict the stations that will be processed.
For example, if only ``--campaign_names`` are specified, DISDRODB will process only the stations of such campaigns.


1. By command line :


	.. code-block::

		run_disdrodb_l0 <disdrodb_dir> --data_sources <data_sources> --campaign_names <campaign_names> --station_names <station_names> [parameters]

	Example :

	.. code-block:: bash

		run_disdrodb_l0 /ltenas8/DISDRODB --campaign_names EPFL_2008 --l0a_processing True --l0b_processing True --parallel False

	To  specify multiple campaigns you can do the follow

	.. code-block:: bash

		run_disdrodb_l0 /ltenas8/DISDRODB --campaign_names 'EPFL_2008 LOCARNO_2018' --l0a_processing True --l0b_processing True --parallel False

     Type ``run_disdrodb_l0 --help`` in the terminal to get more information on the possible parameters.


2. By calling a python function


		.. code-block:: python

			from disdrodb.l0 import run_disdrodb_l0
			run_disdrodb_l0(<disdrodb_dir> <data_source>, <campaign_name>, ...)


		Example :

		.. code-block:: python

			from disdrodb.l0 import run_disdrodb_l0

			disdrodb_dir = "...\\DISDRODB"
			data_sources=['EPFL']
			campaign_names=['EPFL_2008']
			# L0 processing settings
			l0a_processing=True
			l0b_processing=True
			l0b_concat=False
			remove_l0a=False
			remove_l0b=False
			# L0 processing options
			force=True
			verbose=True
			debugging_mode=True
			parallel=False
			l0b_concat=True

			run_disdrodb_l0(
				disdrodb_dir=disdrodb_dir,
				data_sources=data_sources,      # optional
				campaign_names=campaign_names,  # optional
				# station_names=station_names,  # optional
   		     	# L0 processing settings
				l0a_processing=l0a_processing,
				l0b_processing=l0b_processing,
				l0b_concat=l0b_concat,
				remove_l0a=remove_l0a,
				remove_l0b=remove_l0b,
				# L0 processing options
				parallel=parallel,
				verbose=verbose,
				force=force,
				debugging_mode=debugging_mode,
			)
