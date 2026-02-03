.. _processing:

============================
Archive Processing
============================

DISDRODB enables processing of disdrometer data archives from the command line or by calling Python functions.
This guide describes how to generate all DISDRODB products (L0A through L2M) for single stations or entire archives.

For product descriptions, see :ref:`Products <products>`.
For configuration options, see :ref:`Products Configuration <products_configuration>`.

-------------------------------------------------
Processing Chain Overview
-------------------------------------------------

DISDRODB processes raw disdrometer data through a sequential chain:

.. code-block:: text

    Raw Data → L0A → L0B → L0C → L1 → L2E → L2M

**Processing Levels:**

- **L0A**: Standardized tabular data (Apache Parquet format)
- **L0B**: netCDF4 with physical dimensions and CF-compliant metadata
- **L0C**: Time-consistent datasets with fixed measurement intervals
- **L1**: Temporally resampled data with hydrometeor classification
- **L2E**: Empirical rainfall parameters and radar observables
- **L2M**: Modeled DSD parameters from parametric fitting

Each level builds on the previous one. You can process the entire chain or individual levels,
but prerequisite products must exist (e.g., L1 processing requires L0C data).

-------------------------------------------------
Processing Options
-------------------------------------------------

**Common Parameters**

- ``force``: If ``True``, overwrite existing data; if ``False``, raise error if data exists
- ``verbose``: Print detailed processing information to terminal. Only if parallel is ``False``
- ``debugging_mode``: Process only subset of data for testing (3 files for L0A, 100 rows for L0B)
- ``parallel``: Process files simultaneously in multiple processes (recommended for large archives)
- ``data_archive_dir``: Path to DISDRODB Data Archive (if not using default configuration)
- ``metadata_archive_dir``: Path to DISDRODB Metadata Archive (if not using default configuration)

**Station Selection** (for archive-wide processing)

- ``data_sources``: List of data sources to process (e.g., ``["EPFL", "NASA", "ITALY"]``)
- ``campaign_names``: List of campaigns to process (e.g., ``["EPFL_2008", "LOCARNO_2019"]``)
- ``station_names``: List of specific stations to process (e.g., ``["TC-RM", "TC-TO"]``)

If none specified, all available stations are processed. These filters can be combined to select specific subsets.

-------------------------------------------------
L0 Processing (Single Station)
-------------------------------------------------

Process raw data through L0A, L0B, and L0C levels for a specific station.

See :func:`disdrodb.run_l0_station` for detailed parameter documentation.

**Command Line**

.. code-block:: bash

    disdrodb_run_l0_station <data_source> <campaign_name> <station_name> [options]

Example:

.. code-block:: bash

    disdrodb_run_l0_station EPFL EPFL_2008 10 --l0a_processing True --l0b_processing True --l0c_processing True --force True --verbose True --parallel False

Type ``disdrodb_run_l0_station --help`` for all available options.

**Python**

.. code-block:: python

    import disdrodb

    disdrodb.run_l0_station(
        data_source="EPFL",
        campaign_name="EPFL_2008",
        station_name="10",
        # Processing levels
        l0a_processing=True,
        l0b_processing=True,
        l0c_processing=True,
        # Options
        remove_l0a=False,
        remove_l0b=False,
        force=True,
        verbose=True,
        debugging_mode=False,
        parallel=False,
    )

-------------------------------------------------
L0 Processing (Multiple Stations)
-------------------------------------------------

Process multiple stations simultaneously. Filters can be combined to select specific subsets
of the archive.

See :func:`disdrodb.run_l0` for detailed parameter documentation.

**Command Line**

.. code-block:: bash

    disdrodb_run_l0 --data_sources <sources> --campaign_names <campaigns> --station_names <stations> [options]

Example - Process entire campaign:

.. code-block:: bash

    disdrodb_run_l0 --campaign_names EPFL_2008 --l0a_processing True --l0b_processing True --l0c_processing True --parallel False

Example - Process multiple campaigns:

.. code-block:: bash

    disdrodb_run_l0 --campaign_names 'EPFL_2008 LOCARNO_2018' --l0a_processing True --l0b_processing True --l0c_processing True --parallel True

Type ``disdrodb_run_l0 --help`` for all available options.

**Python**

.. code-block:: python

    import disdrodb

    disdrodb.run_l0(
        data_sources=["EPFL"],
        campaign_names=["EPFL_2008"],
        # station_names=["10", "20"],  # Optional: specific stations only
        # Processing levels
        l0a_processing=True,
        l0b_processing=True,
        l0c_processing=True,
        # Options
        remove_l0a=False,
        remove_l0b=False,
        force=True,
        verbose=True,
        debugging_mode=False,
        parallel=True,
    )

-------------------------------------------------
L1 Processing (Single Station)
-------------------------------------------------

Generate temporally resampled data with hydrometeor classification from L0C products.

See :func:`disdrodb.run_l1_station` for detailed argument documentation.

**Command Line**

.. code-block:: bash

    disdrodb_run_l1_station <data_source> <campaign_name> <station_name> [options]

Example:

.. code-block:: bash

    disdrodb_run_l1_station EPFL EPFL_2008 10 --force True --verbose True --parallel True

**Python**

.. code-block:: python

    import disdrodb

    disdrodb.run_l1_station(
        data_source="EPFL",
        campaign_name="EPFL_2008",
        station_name="10",
        force=True,
        verbose=True,
        debugging_mode=False,
        parallel=True,
    )

-------------------------------------------------
L1 Processing (Multiple Stations)
-------------------------------------------------

Process L1 products for multiple stations.

See :func:`disdrodb.run_l1` for detailed argument documentation.

**Command Line**

.. code-block:: bash

    disdrodb_run_l1 --campaign_names <campaigns> [options]

Example:

.. code-block:: bash

    disdrodb_run_l1 --campaign_names EPFL_2008 --force True --parallel True

**Python**

.. code-block:: python

    import disdrodb

    disdrodb.run_l1(
        campaign_names=["EPFL_2008"],
        force=True,
        verbose=True,
        parallel=True,
    )

-------------------------------------------------
L2E Processing (Single Station)
-------------------------------------------------

Compute integrated rainfall parameters from L1 products.

See :func:`disdrodb.run_l2e_station` for detailed argument documentation.

**Command Line**

.. code-block:: bash

    disdrodb_run_l2e_station <data_source> <campaign_name> <station_name> [options]

Example:

.. code-block:: bash

    disdrodb_run_l2e_station EPFL EPFL_2008 10 --force True --verbose True --parallel True

**Python**

.. code-block:: python

    import disdrodb

    disdrodb.run_l2e_station(
        data_source="EPFL",
        campaign_name="EPFL_2008",
        station_name="10",
        force=True,
        verbose=True,
        debugging_mode=False,
        parallel=True,
    )

-------------------------------------------------
L2E Processing (Multiple Stations)
-------------------------------------------------

Process L2E products for multiple stations.

See :func:`disdrodb.run_l2e` for detailed argument documentation.

**Command Line**

.. code-block:: bash

    disdrodb_run_l2e --campaign_names <campaigns> [options]

Example:

.. code-block:: bash

    disdrodb_run_l2e --campaign_names EPFL_2008 --force True --parallel True

**Python**

.. code-block:: python

    import disdrodb

    disdrodb.run_l2e(
        campaign_names=["EPFL_2008"],
        force=True,
        verbose=True,
        parallel=True,
    )

-------------------------------------------------
L2M Processing (Single Station)
-------------------------------------------------

Fit parametric DSD models to the drop number concentration derived in L2E products.

See :func:`disdrodb.run_l2m_station` for detailed argument documentation.

**Command Line**

.. code-block:: bash

    disdrodb_run_l2m_station <data_source> <campaign_name> <station_name> [options]

Example:

.. code-block:: bash

    disdrodb_run_l2m_station EPFL EPFL_2008 10 --force True --verbose True --parallel True

**Python**

.. code-block:: python

    import disdrodb

    disdrodb.run_l2m_station(
        data_source="EPFL",
        campaign_name="EPFL_2008",
        station_name="10",
        force=True,
        verbose=True,
        debugging_mode=False,
        parallel=True,
    )

-------------------------------------------------
L2M Processing (Multiple Stations)
-------------------------------------------------

Process L2M products for multiple stations.

See :func:`disdrodb.run_l2m` for detailed argument documentation.

**Command Line**

.. code-block:: bash

    disdrodb_run_l2m --campaign_names <campaigns> [options]

Example:

.. code-block:: bash

    disdrodb_run_l2m --campaign_names EPFL_2008 --force True --parallel True

**Python**

.. code-block:: python

    import disdrodb

    disdrodb.run_l2m(
        campaign_names=["EPFL_2008"],
        force=True,
        verbose=True,
        parallel=True,
    )

-------------------------------------------------
Complete Processing Chain
-------------------------------------------------

Process entire chain from raw data to final products in one command.
By default, L2M processing is is disabled. Enable with ``l2m_processing=True``.

**Single Station**

See :func:`disdrodb.run_station` for detailed argument documentation.

.. code-block:: python

    import disdrodb

    disdrodb.run_station(
        data_source="EPFL",
        campaign_name="EPFL_2008",
        station_name="10",
        # L0 processing
        l0a_processing=True,
        l0b_processing=True,
        l0c_processing=True,
        remove_l0a=False,
        remove_l0b=False,
        # L1 and L2 processing
        l1_processing=True,
        l2e_processing=True,
        l2m_processing=False,
        # Options
        force=True,
        verbose=True,
        debugging_mode=False,
        parallel=True,
    )

**Multiple Stations**

See :func:`disdrodb.run` for detailed parameter documentation.

.. code-block:: python

    import disdrodb

    disdrodb.run(
        campaign_names=["EPFL_2008"],
        # L0 processing
        l0a_processing=True,
        l0b_processing=True,
        l0c_processing=True,
        remove_l0a=False,
        remove_l0b=False,
        # L1 and L2 processing
        l1_processing=True,
        l2e_processing=True,
        l2m_processing=False,
        # Options
        force=True,
        verbose=True,
        debugging_mode=False,
        parallel=True,
    )

-------------------------------------------------
Best Practices
-------------------------------------------------

**Processing Strategy**

1. **Start Small**: Test with ``debugging_mode=True`` on a single station before processing entire archives
2. **Sequential Testing**: Process one product level at a time initially to verify configurations
3. **Parallel Processing**: Enable ``parallel=True`` for large archives to speed up processing
4. ** Memory Usage**: Monitor memory usage when processing multiple stations in parallel; adjust archive options if necessary
5. **Disk Space**: Monitor disk space, especially when processing L0 products without removing intermediate files
6. **Configuration**: Customize products configurations before large-scale processing (see :ref:`Products Configuration <products_configuration>`)

**Error Handling**

- Use ``force=False`` to prevent accidental overwriting of existing data
- Use ``verbose=True`` during initial testing to monitor progress
- Check logs directory for detailed error messages.


**Memory Management**

- Adjust ``parallel`` settings based on available memory
- Consider processing in batches for very large archives
- Use ``remove_l0a=True`` and ``remove_l0b=True`` to save disk space after L0C generation

-------------------------------------------------
Command Reference
-------------------------------------------------

**Single Station Commands**

- ``disdrodb_run_l0_station``: L0A → L0B → L0C
- ``disdrodb_run_l0a_station``: Raw → L0A
- ``disdrodb_run_l0b_station``: L0A → L0B
- ``disdrodb_run_l0c_station``: L0B → L0C
- ``disdrodb_run_l1_station``: L0C → L1
- ``disdrodb_run_l2e_station``: L1 → L2E
- ``disdrodb_run_l2m_station``: L2E → L2M
- ``disdrodb_run_station``: Raw → all products

**Archive-Wide Commands**

- ``disdrodb_run_l0``: L0 processing for multiple stations
- ``disdrodb_run_l0a``: L0A processing for multiple stations
- ``disdrodb_run_l0b``: L0B processing for multiple stations
- ``disdrodb_run_l0c``: L0C processing for multiple stations
- ``disdrodb_run_l1``: L1 processing for multiple stations
- ``disdrodb_run_l2e``: L2E processing for multiple stations
- ``disdrodb_run_l2m``: L2M processing for multiple stations
- ``disdrodb_run``: Complete chain for multiple stations

Use ``<command> --help`` for detailed parameter information.

By typing ``disdrodb`` followed by TAB TAB TAB in the terminal, you can see all available DISDRODB commands.
