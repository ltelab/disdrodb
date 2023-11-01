# 📦 DISDRODB - A package to standardize, process and analyze global disdrometer data.


.. |pypi| image:: https://badge.fury.io/py/disdrodb.svg
   :target: https://pypi.org/project/disdrodb/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/disdrodb.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/disdrodb

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/disdrodb.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/disdrodb/

.. |conda_downloads| image:: https://img.shields.io/conda/dn/conda-forge/disdrodb.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/disdrodb

.. |versions| image:: https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11|%203.12-blue
   :target: https://www.python.org/downloads/
   :alt: Supported Python Versions

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

.. |slack| image:: https://img.shields.io/badge/Slack-disdrodb-green.svg?logo=slack
   :target: https://disdrodbworkspace.slack.com/

.. |discussion| image:: https://img.shields.io/badge/GitHub-Discussions-green?logo=github
   :target: https://github.com/ltelab/disdrodb/discussions

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: ruff

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
   :target: https://github.com/psf/black
   :alt: black

.. |codespell| image:: https://img.shields.io/badge/Codespell-enabled-brightgreen
  :target: https://github.com/codespell-project/codespell
  :alt: codespell

.. |openssf| image:: https://www.bestpractices.dev/projects/XXXX/badge
  :target: https://www.bestpractices.dev/projects/XXXX
  :alt: OpenSSF Best Practices

.. |pyopensci| image:: https://tinyurl.com/XXXX
   :target: https://github.com/pyOpenSci/software-review/issues/XXX
   :alt: pyOpenSci

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
| Linting              | |black|  |ruff|  |codespell|                |
+----------------------+---------------------------------------------+
| Code Coverage        | |coverall| |codecov|                        |
+----------------------+---------------------------------------------+
| Code Quality         | |codefactor| |codebeat|                     |
|                      +---------------------------------------------+
|                      | |codacy| |codescene|                        |
+----------------------+---------------------------------------------+
| Code Review          | |pyopensci| |openssf|                       |
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
DISDRODB standards are being established following FAIR data best practices and Climate & Forecast (CF) conventions, and will facilitate
the preprocessing, analysis and visualization of disdrometer data.

## ℹ️ Software Overview

The software currently enable to:
- download the raw disdrometer data from all stations included in the DISDRODB Decentralized Data Archive
- upload raw disdrometer data from the user to the DISDRODB Decentralized Data Archive
- process more than 400 disdrometer stations into a standard NetCDF format (DISDRODB L0 product)

Currently, the DISDRODB Working Group is discussing the development of various scientific products. :
If you have ideas, algorithms, data or expertise to share, or you want to contribute to the future DISDRODB products, do not hesitate to  **GET IN TOUCH** !!!

Join the [**DISDRODB Slack Workspace**](http://slack.disdrodb.org) to meet the DISDRODB Community !


## 🚀 Quick Start

You're about to create your very own DISDRODB Data Archive. All it takes is a simple command-line journey to your chosen directory.

#### 📚 Set up the DISDRODB Metadata And Local Data Archive

Let's start by travel to the directory where you want to store the DISDRODB Data Archive with :code:`cd <your_desired_root_folder>`.

Then clone the DISDRODB Metadata Archive repository with:

.. code:: bash

   git clone https://github.com/ltelab/disdrodb-data.git

This will create a directory called ``disdrodb-data``, which is ready to be filled with data from the DISDRODB Decentralized Data Archive.

But before starting to download some data, we need to specify the location of the DISDRODB Local Archive.

You can specify once for ever the default DISDRODB Local Archive directory by running in python:

```python
   import disdrodb
   disdrodb_dir = "<the_root_folder>/disdrodb-data/DISDRODB>"
   disdrodb.define_configs(disdrodb_dir=disdrodb_dir)
```

or set up a (temporary) environment variable `DISDRODB_DIR` in your terminal with:

```bash
   export DISDRODB_DIR="<the_root_folder>/disdrodb-data/DISDRODB>"
```

#### 📥 Download the raw data of the DISDRODB stations

To download all data stored into the DISDRODB Decentralized Data Archive, you just have to run the following command:

```bash
   download_disdrodb_archive
```

#### 💫 Transform the raw data to standardized netCDF files (DISDRODB L0 product).

Then, if you want to convert all stations raw data into standardized netCDF4 files, run the following command in the terminal:

```bash

   run_disdrodb_l0

```

#### 📖 Explore the DISDRODB documentation

To discover all download and processing options, or how to contribute your own data to DISDRODB,
please read the software documentation available at [https://disdrodb.readthedocs.io/en/latest/](https://disdrodb.readthedocs.io/en/latest/).

If you want to improve to the DISDRODB Metadata Archive repository, you can explore the repository
at [https://github.com/ltelab/disdrodb-data](https://github.com/ltelab/disdrodb-data)


## 🛠️ Installation


DISDRODB can be installed from PyPI with pip:

  ```bash

  pip install disdrodb

  ```

## 💭 Feedback and Contributing Guidelines

If you aim to contribute your data or discuss the future development of DISDRODB,
we highly suggest to join the [**DISDRODB Slack Workspace**](http://slack.disdrodb.org)

Feel free to also open a [GitHub Issue](https://github.com/ltelab/disdrodb/issues) or a
[GitHub Discussion](https://github.com/ltelab/disdrodb/discussions) specific to your questions or ideas.


## ✍️  Contributors

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
