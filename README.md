# DISDRODB - A package to standardize, process and analyze global disdrometer data.

[![DOI](https://zenodo.org/badge/DOI/XXX)](https://doi.org/10.5281/zenodo.XXXX)
[![PyPI version](https://badge.fury.io/py/disdrodb.svg)](https://badge.fury.io/py/disdrodb)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/disdrodb.svg)](https://anaconda.org/conda-forge/disdrodb)
[![Build Status](https://github.com/ltelab/disdrodb/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/ltelab/disdrodb/actions)
[![Coverage Status](https://coveralls.io/repos/github/ltelab/disdrodb/badge.svg?branch=main)](https://coveralls.io/github/ltelab/disdrodb?branch=main)
[![Documentation Status](https://readthedocs.org/projects/disdrodb/badge/?version=latest)](https://disdrodb.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/github/license/ltelab/disdrodb)](https://github.com/ltelab/disdrodb/blob/master/LICENSE)

DISDRODB is part of an initial effort to index, collect and homogenize drop size distribution (DSD) data sets across the globe,
as well as to establish a global standard for disdrometers observations data sharing. 
DISDRODB standards are being established following FAIR data best practices and Climate & Forecast (CF) conventions, and will facilitate the preprocessing, analysis and visualization of disdrometer data.  

The DISDRODB archive is composed of 3 product levels:
- L0 provides the raw sensors measurements converted into a standardized netCDF4 format.
- L1 provides L0 homogenized and quality-checked data.
- L2 provides scientific products derived from the L1 data.

The code required to the generate the DISDRODB archive is enclosed in the `production` directory of the repository. 

The code facilitating the analysis and visualization of the DISDRODB archive is available in the `api` directory.


The software documentation is available at [https://disdrodb.readthedocs.io/en/latest/](https://disdrodb.readthedocs.io/en/latest/). 

Currently: 
- only the DISDRODB L0 product generation has been implemented;
- the pipeline for DISDRODB L1 and L2 product generation is in development;
- the DISDRODB API is in development; 
- more than 300 sensors have been already processed to DISDRODB L0; 
- tens of institutions have manifested their interest in adopting the DISDRODB tools and standards. 

Consequently **IT IS TIME TO GET INVOLVED**. If you have ideas, algorithms, data or expertise to share, do not hesitate to **GET IN TOUCH** !!!




## Installation


DISDRODB can be installed from PyPI with pip: 

  ```sh
  pip install disdrodb
  ```
  
In future, it will become available from conda-forge for conda installations:   

  ```sh
  conda install -c conda-forge disdrodb
  ```
 
## Contributors

* [Gionata Ghiggi](https://people.epfl.ch/gionata.ghiggi)
* [Kim Candolfi](https://github.com/KimCandolfi)
* [Régis Longchamp](https://people.epfl.ch/regis.longchamp)
* [Charlotte Gisèle Weil](https://people.epfl.ch/charlotte.weil)
* [Jacopo Grazioli](https://people.epfl.ch/jacopo.grazioli) 
* [Alexis Berne](https://people.epfl.ch/alexis.berne?lang=en)

## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
