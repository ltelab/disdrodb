Welcome to DISDRODB!
======================


Motivation
===========

The raindrop size distribution (DSD) describes the concentration and size
distributions of raindrops in a volume of air.
It is crucial for:

- modeling the propagation of microwave signals through the atmosphere (key for telecommunication and weather radar remote sensing calibration),
- improving microphysical schemes in numerical weather prediction models,
- understanding land surface processes (rainfall interception, soil erosion).

The need for understanding the DSD spatio-temporal variability has led scientists
all around the globe to "count the drops" by deploying DSD recording instruments
known as disdrometers.
Numerous measurement campaigns have been run by various meteorological services,
national agencies (NASA, NCAR, ARM, ...), and university research groups.
However, only a small fraction of this data is easily accessible.
Data are stored in disparate formats with poor documentation, making them difficult to share, analyze, compare, and reuse.
Additionally, very limited software exists or is publicly available for DSD processing.

In response to these challenges, the **disdrodb** software provides a
set of tools to download, process, and archive disdrometer data
following best open science practices.

The goals of the DISDRODB initiative are to:

- create a decentralized archive of disdrometer data from around the world,
- promote data exchange across the scientific community,
- document the available data and types of disdrometer sensors,
- provide a common framework for processing disdrometer data,
- develop a set of scientific products to study DSD spatio-temporal variability at global scale,
- create a community to develop, share, and improve DSD models and algorithms.


Software
===========

The software currently enables you to:

- download raw disdrometer data from stations in the DISDRODB Decentralized Data Archive,
- upload raw disdrometer data to the DISDRODB Decentralized Data Archive,
- read raw measurements from over 400 disdrometer stations and save them in a standard NetCDF format (DISDRODB L0 product).

The disdrodb software can ingest data from various disdrometer sensors and manufacturers,
and is designed so that anyone can easily add support for new instruments.

Currently, disdrodb enables processing of data acquired from the following disdrometer sensors:

- Distromet RD-80 (``RD80``),
- OTT Parsivel (``PARSIVEL``),
- OTT Parsivel2 (``PARSIVEL2``),
- Thies Laser Precipitation Monitor (``LPM``),
- Campbell Present Weather Sensor 100 (``PWS100``),
- Eigenbrot Optical Disdrometer 470 (``ODM470``),
- Biral Visibility and Present Weather Sensors (``SWS250``).

If you have data from other disdrometer types and would like to contribute to the DISDRODB project,
we welcome your help. We're especially interested in adding support for:

- Vaisala Forward Scatter Sensor FD70 (``FD70``),
- Biral Visibility and Present Weather Sensors (``VPF730``, ``VPF750``),
- Joanneum Research Two Dimensional Video Disdrometer (``2DVD``).


Data Archive
==============

The DISDRODB Decentralized Data Archive collects disdrometer data from around the world.

The data are stored in remote repositories but are easily accessible through the disdrodb software.

The metadata for each station is stored in a `centralized repository hosted on GitHub <https://github.com/ltelab/DISDRODB-METADATA>`__.

Explore the currently available disdrometer stations on the map below.

Click the map preview to open the full-screen interactive DISDRODB station map.

.. image:: https://raw.githubusercontent.com/ltelab/DISDRODB-METADATA/main/.github/map_preview.png
   :alt: Open the interactive map
   :target: https://ltelab.github.io/disdrodb-webmap/
   :width: 100%
   :align: center


.. warning::

    Users are expected to properly acknowledge the data they use by citing
    and referencing each station. The corresponding references, recommended
    citations, and DOIs are available in the DISDRODB NetCDF/xarray.Dataset
    global attributes and in the DISDRODB Metadata Archive.


Community
===========

The DISDRODB Working Group is an open community of scientists and engineers interested in advancing the DISDRODB initiative.

We are currently finalizing the DISDRODB L1 product,
featuring quality-checked disdrometer data, along with a suite of research-oriented DISDRODB L2 products.

Your ideas, algorithms, data, and expertise could significantly shape the future of DISDRODB products,
and we would be happy to have you as part of this collaborative project.

If you are eager to contribute or simply curious about what we do, please reach out.

Join the `DISDRODB Slack Workspace <https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA>`__
and say hi!


Documentation
=============

.. toctree::
   :maxdepth: 2

   installation
   quick_start
   products
   products_configuration
   archive_processing
   advanced_tutorials
   metadata_archive
   metadata
   readers
   sensor_configs
   contribute_data
   contributors_guidelines
   maintainers_guidelines
   software_structure
   authors

.. toctree::
   :maxdepth: 2

   DISDRODB API <api/modules>





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
