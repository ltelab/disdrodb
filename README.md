# 📦 DISDRODB - A package to standardize, process and analyze global disdrometer data.

|                      |                                                |
| -------------------- | ---------------------------------------------- |
| Deployment           | [![PyPI](https://badge.fury.io/py/disdrodb.svg?style=flat)](https://pypi.org/project/disdrodb/) [![Conda](https://img.shields.io/conda/vn/conda-forge/disdrodb.svg?logo=conda-forge&logoColor=white&style=flat)](https://anaconda.org/conda-forge/disdrodb) |
| Activity             | [![PyPI Downloads](https://img.shields.io/pypi/dm/disdrodb.svg?label=PyPI%20downloads&style=flat)](https://pypi.org/project/disdrodb/) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/disdrodb.svg?label=Conda%20downloads&style=flat)](https://anaconda.org/conda-forge/disdrodb) |
| Python Versions      | [![Python Versions](https://img.shields.io/badge/Python-3.8%20%203.9%20%203.10%20%203.11%20%203.12-blue?style=flat)](https://www.python.org/downloads/) |
| Supported Systems    | [![Linux](https://img.shields.io/github/actions/workflow/status/ltelab/disdrodb/.github/workflows/tests.yml?label=Linux&style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/tests.yml) [![macOS](https://img.shields.io/github/actions/workflow/status/ltelab/disdrodb/.github/workflows/tests.yml?label=macOS&style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/tests.yml) [![Windows](https://img.shields.io/github/actions/workflow/status/ltelab/disdrodb/.github/workflows/tests_windows.yml?label=Windows&style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/tests_windows.yml) |
| Project Status       | [![Project Status](https://www.repostatus.org/badges/latest/active.svg?style=flat)](https://www.repostatus.org/#active) |
| Build Status         | [![Tests](https://github.com/ltelab/disdrodb/actions/workflows/tests.yml/badge.svg?style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/tests.yml) [![Lint](https://github.com/ltelab/disdrodb/actions/workflows/lint.yml/badge.svg?style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/lint.yml) [![Docs](https://readthedocs.org/projects/disdrodb/badge/?version=latest&style=flat)](https://disdrodb.readthedocs.io/en/latest/) |
| Linting              | [![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat)](https://github.com/astral-sh/ruff) [![Codespell](https://img.shields.io/badge/Codespell-enabled-brightgreen?style=flat)](https://github.com/codespell-project/codespell) |
| Code Coverage        | [![Coveralls](https://coveralls.io/repos/github/ltelab/disdrodb/badge.svg?branch=main&style=flat)](https://coveralls.io/github/ltelab/disdrodb?branch=main) [![Codecov](https://codecov.io/gh/ltelab/disdrodb/branch/main/graph/badge.svg?style=flat)](https://codecov.io/gh/ltelab/disdrodb) |
| Code Quality         | [![Codefactor](https://www.codefactor.io/repository/github/ltelab/disdrodb/badge?style=flat)](https://www.codefactor.io/repository/github/ltelab/disdrodb) [![Codebeat](https://codebeat.co/badges/14ff831b-f064-4bdd-a2e2-72ffdf28a35a?style=flat)](https://codebeat.co/projects/github-com-ltelab-disdrodb-main) [![Codacy](https://app.codacy.com/project/badge/Grade/d823c50a7ad14268bd347b5aba384623?style=flat)](https://app.codacy.com/gh/ltelab/disdrodb/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Codescene](https://codescene.io/projects/36773/status-badges/code-health?style=flat)](https://codescene.io/projects/36773) |
| Code Review          | [![pyOpenSci](https://tinyurl.com/XXXX)](#) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/XXXX/badge?style=flat)](#) |
| License              | [![License](https://img.shields.io/github/license/ltelab/disdrodb?style=flat)](https://github.com/ltelab/disdrodb/blob/main/LICENSE) |
| Community            | [![Slack](https://img.shields.io/badge/Slack-disdrodb-green.svg?logo=slack&style=flat)](https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA) [![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-green?logo=github&style=flat)](https://github.com/ltelab/disdrodb/discussions) |
| Citation             | [![JOSS](http://joss.theoj.org/papers/<DOI>/joss.<DOI>/status.svg?style=flat)](#) [![DOI](https://zenodo.org/badge/429018433.svg?style=flat)](#) |

 [**Slack**](https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA) | [**Docs**](https://disdrodb.readthedocs.io/en/latest/)

DISDRODB is part of an initial effort to index, collect and homogenize drop size distribution (DSD) data sets across the globe,
as well as to establish a global standard for disdrometers observations data sharing.
DISDRODB standards are being established following FAIR data best practices and Climate & Forecast (CF) conventions, and will facilitate
the preprocessing, analysis and visualization of disdrometer data.

## ℹ️ Software Overview

The software currently enable to:
- download the raw disdrometer data from all stations included in the DISDRODB Decentralized Data Archive
- upload raw disdrometer data from the user to the DISDRODB Decentralized Data Archive
- process more than 400 disdrometer stations into a standard NetCDF format (DISDRODB L0 product)

Currently, the DISDRODB Working Group is discussing the development of various scientific products.
If you have ideas, algorithms, data or expertise to share, or you want to contribute to the future DISDRODB products, do not hesitate to get in touch !!!

Join the [**DISDRODB Slack Workspace**](https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA) to meet the DISDRODB Community !


## 🚀 Quick Start

You're about to create your very own DISDRODB Local Data Archive. All it takes is a simple command-line journey.

#### 📚 Set up the DISDRODB Metadata And Local Data Archive

Let's start by travel to the directory where you want to store the DISDRODB Data Archive.

Then clone the DISDRODB Metadata Archive repository with:

```bash

   git clone https://github.com/ltelab/disdrodb-data.git
```

This will create a directory called ``disdrodb-data``, which is ready to be filled with data from the DISDRODB Decentralized Data Archive.

But before starting to download some data, we need to specify the location of the DISDRODB Local Archive.

You can specify once forever the default DISDRODB Local Archive directory by running in python:

```python
   import disdrodb
   base_dir = "<the_path_to>/disdrodb-data/DISDRODB>"
   disdrodb.define_configs(base_dir=base_dir)
```

or set up the (temporary) environment variable `DISDRODB_BASE_DIR` in your terminal with:

```bash
   export DISDRODB_BASE_DIR="<the_path_to>/disdrodb-data/DISDRODB>"
```

#### 📥 Download the DISDRODB raw data

To download all data stored into the DISDRODB Decentralized Data Archive, you just have to run the following command:

```bash
   disdrodb_download_archive
```

#### 💫 Transform the raw data to standardized netCDF files.

If you want to convert all stations raw data into standardized netCDF4 files, run the following command in the terminal:

```bash

   disdrodb_run_l0

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
we highly suggest to join the [**DISDRODB Slack Workspace**](https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA)

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
