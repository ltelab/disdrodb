# DISDRODB - A package to standardize, process and analyze global disdrometer data.


.. |pypi| image:: https://badge.fury.io/py/disdrodb.svg
   :target: https://pypi.org/project/disdrodb/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/disdrodb.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/disdrodb


.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/disdrodb.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/disdrodb/

.. |conda_downloads| image:: https://img.shields.io/conda/dn/conda-forge/disdrodb.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/disdrodb

# TODO PYTHON VERSIONS
[![image](https://img.shields.io/pypi/pyversions/ruff.svg)](https://pypi.python.org/pypi/ruff)
.. |python| image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active


.. |tests| image:: https://github.com/ltelab/disdrodb/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/ltelab/disdrodb/actions/workflows/tests.yml

.. |lint| image:: https://github.com/ltelab/disdrodb/actions/workflows/lint.yml/badge.svg
   :target: https://github.com/ltelab/disdrodb/actions/workflows/lint.yml

.. |docs| image:: https://readthedocs.org/projects/disdrodb/badge/?version=latest
   :target: https://disdrodb.readthedocs.io/en/latest/

.. |coverall| image:: https://coveralls.io/repos/github/ltelab/disdrodb/badge.svg?branch=main
   :target: https://coveralls.io/github/ltelab/disdrodb?branch=main

.. |codecov| image:: https://codecov.io/gh/ltelab/disdrodb/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/ltelab/disdrodb

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/d823c50a7ad14268bd347b5aba384623
   :target: https://app.codacy.com/gh/ltelab/disdrodb/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade

.. |codescene| image:: https://codescene.io/projects/36773/status-badges/code-health
   :target: https://codescene.io/projects/36773

.. |codefactor| image:: https://www.codefactor.io/repository/github/ltelab/disdrodb/badge
   :target: https://www.codefactor.io/repository/github/ltelab/disdrodb

.. |codebeat| image:: https://codebeat.co/badges/14ff831b-f064-4bdd-a2e2-72ffdf28a35a
   :target: https://codebeat.co/projects/github-com-ltelab-disdrodb-main

.. |licence| image:: https://img.shields.io/github/license/ltelab/disdrodb
   :target: https://github.com/ltelab/disdrodb/blob/main/LICENSE

# TODO slack link and badge
.. |slack| image:: https://img.shields.io/badge/Slack-disdrodb-green.svg?logo=slack
   :target: http://slack.disdrodb.org

.. |discussion| image:: https://img.shields.io/badge/GitHub-Discussions-green?logo=github
   :target: https://github.com/ltelab/disdrodb/discussions

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
  :target: https://github.com/astral-sh/ruff
  :alt: ruff

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
  :target: https://github.com/psf/black
  :alt: black

# TODO badge
.. |codespell| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
  :target: https://github.com/codespell-project/codespell
  :alt: codespell

.. |joss| image:: http://joss.theoj.org/papers/<DOI>/joss.<DOI>/status.svg
   :target: https://doi.org/

.. |DOI| image:: https://zenodo.org/badge/429018433.svg
   :target: https://zenodo.org/badge/latestdoi/429018433


+----------------------+------------------------+--------------------+
| Deployment           | |pypi|                 | |conda|            |
+----------------------+------------------------+--------------------+
| Activity             | |pypi_downloads|       | |conda_downloads|  |
+----------------------+------------------------+--------------------+
| Python versions      | |python|                                    |
+----------------------+---------------------------------------------+
| Project status       | |status|                                    |
+----------------------+---------------------------------------------+
| Build Status         | |tests| |lint| |docs|                       |
+----------------------+---------------------------------------------+
| Code Coverage        | |coverall| |codecov|                        |
+----------------------+---------------------------------------------+
| Code Quality         |codefactor| |codebeat|                       |
|                      +---------------------------------------------+
|                      | |codacy| |codescene|                        |
+----------------------+---------------------------------------------+
| Linting              | |black|  |ruff|  |codespell|                |
+----------------------+---------------------------------------------+
| License              | |licence|                                   |
+----------------------+------------------------+--------------------+
| Community            | |slack|                | |discussion|       |
+----------------------+------------------------+--------------------+
| Citation             | |joss|                 | |DOI|              |
+----------------------+------------------------+--------------------+

 [**Slack**](http://slack.disdrodb.org) | [**Docs**](https://disdrodb.readthedocs.io/en/latest/)


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

## Contributors

* [Gionata Ghiggi](https://people.epfl.ch/gionata.ghiggi)
* [Kim Candolfi](https://github.com/KimCandolfi)
* [Régis Longchamp](https://people.epfl.ch/regis.longchamp)
* [Charlotte Gisèle Weil](https://people.epfl.ch/charlotte.weil)
* [Jacopo Grazioli](https://people.epfl.ch/jacopo.grazioli)
* [Alexis Berne](https://people.epfl.ch/alexis.berne?lang=en)

## Citation

You can cite the DISDRODB software by:

> Gionata Ghiggi, Kim Candolfi, Régis Longchamp, Charlotte Weil, Alexis Berne (2023). ltelab/disdrodb  Zenodo. https://doi.org/10.5281/zenodo.7680581

If you want to cite a specific version, have a look at the [Zenodo site](https://doi.org/10.5281/zenodo.7680581)

## License

The content of this repository is released under the terms of the [GPL 3.0 license](LICENSE).
