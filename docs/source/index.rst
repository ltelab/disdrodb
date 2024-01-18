.. d documentation master file, created by
   sphinx-quickstart on Wed Jul 13 14:44:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DISDRODB's documentation!  
=======================================

.. image:: /static/logo.png
   :width: 100%
   :align: center

Motivation
===========

The raindrop size distribution (DSD) describes the concentration and size
distributions of raindrops in a volume of air.
It is a crucial piece of  information to model the propagation of microwave signals
through the atmosphere (key for telecommunication and weather radar remote sensing calibration),
to improve microphysical schemes in numerical weather prediction models, and
to understand land surface processes (rainfall interception, soil erosion).

The need for understanding the DSD spatio-temporal variability has led scientists
all around the globe to "count the drops" by deploying DSD recording instruments
known as disdrometers.
Numerous measurement campaigns have been run by various meteorological services,
national agencies (NASA, NCAR, ARM, ...), and university research groups.
However, only a small fraction of those data is easily accessible.
Data are stored in disparate formats with poor documentation, making them difficult to share, analyse, compare and reuse.
Additionally, very limited software exists or is publicly available for DSD processing.

In response to these challenges, the disdrodb Python package provides a
set of tools to download, process and archive disdrometer data
following the best open science practices.

The goal of the DISDRODB initiative is to:

- create a decentralized archive of disdrometer data from all around the world
- promote the exchange of data across the scientific community
- document the available data and the type of disdrometer sensors
- provide a common framework to process disdrometer data
- develop a set of scientific products to study the DSD variability at various spatio-temporal scales
- create a community to develop, share and improve algorithms


Software
===========

The software currently enable to:

- download the raw disdrometer data from stations included in the DISDRODB Decentralized Data Archive
- upload raw disdrometer data from the user to the DISDRODB Decentralized Data Archive
- read the raw measurements and convert more than 400 disdrometer stations into a standard NetCDF format (DISDRODB L0 product)

The disdrodb software is able to process data from various disdrometer sensors and manufacturers, and is designed to be easily extended to new ones.
Currently, disdrodb enables to process data acquired from:

- the OTT Parsivel (``OTT_Parsivel``),
- the OTT Parsivel2 (``OTT_Parsivel2``),
- the Thies Laser Precipitation Monitor (``Thies_LPM``),
- the RD-80 (``RD_80``) disdrometer.

Data Archive
==============

The DISDRODB Decentralized Data Archive is a collection of disdrometer data from all around the world.

The data are stored in remote data repositories, and are easily accessible through the disdrodb software.

The metadata of each stations are stored in a `centralized repository hosted on GitHub <https://github.com/ltelab/disdrodb-data>`__.

The available disdrometer stations are depicted in the figure below.

.. image:: /static/map_stations.png
   :width: 100%
   :alt: Location of the disdrometer stations included in the DISDRODB Decentralized Data Archive
   :align: center


Community
===========

The DISDRODB Working Group is a open community of scientists and engineers interested in advancing the DISDRODB initiative.

We are currently planning the development of the DISDRODB L1 product,
featuring quality-checked disdrometer data, along with a suite of scientific-research-oriented DISDRODB L2 products.

Your ideas, algorithms, data, and expertise could significantly shape the future of DISDRODB products,
and we would absolutely love for you to be part of this collaborative project.

If you are eager to contribute or simply curious about what we do, please do not hesitate to reach out.

Feel warmly invited to join the `DISDRODB Slack Workspace <https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA>`__
and say hi !


Documentation
=============

.. toctree::
   :maxdepth: 2

   installation
   data_download
   l0_processing
   metadata_archive
   readers
   metadata
   sensor_configs
   contribute_data
   contributors_guidelines
   maintainers_guidelines
   authors
   software_structure


.. toctree::
   :maxdepth: 2

   DISDRODB API <api/modules>





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
