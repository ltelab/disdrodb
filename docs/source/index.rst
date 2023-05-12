.. d documentation master file, created by
   sphinx-quickstart on Wed Jul 13 14:44:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to disdrodb's documentation!
=======================================

DISDRODB: A global database of raindrop size distribution observations


Motivation
================

The raindrop size distribution (DSD) describes the concentration and size
distributions of raindrops in a volume of air.
It is a crucial piece of  information to model the propagation of microwave signals
through the atmosphere (key for telecommunication and weather radar remote sensing calibration),
to improve microphysical schemes in numerical weather prediction models, and
to understand land surface processes (rainfall interception, soil erosion).

The need for understanding the DSD spatio-temporal variability has led scientists
all around the globe to “count the drops” by deploying DSD recording instruments
known as disdrometers.
Numerous measurement campaigns have been run by various meteorological services,
national agencies (e.g. the NASA Precipitation Measurement Mission - PMM - Science Team),
and university research groups.
However, only a small fraction of those data is easily accessible.
Data are stored in disparate formats with poor documentation, making them
difficult to share, analyse, compare and re-use.

Additionally, very limited software is currently publicly available for DSD processing.

This software aims to define a standard format to save disdrometer data and to create a decentralized archive to promote the exchange of data across the scientific community.
Currently, disdrodb enables to process data acquired from the OTT Parsivel (``OTT_Parsivel``), OTT Parsivel2 (``OTT_Parsivel2``), Thies Laser Precipitation Monitor (``ThiesLPM``) and RD-80 (``RD_80``) disdrometers.


Documentation
=============

.. toctree::
   :maxdepth: 2

   installation
   overview
   data
   metadata
   readers
   sensor_configs
   l0_processing
   contributors_guidelines
   maintainers_guidelines
   authors



.. toctree::
   :maxdepth: 1

   disdrodb API <api/modules>





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
