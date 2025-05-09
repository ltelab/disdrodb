# 📦 DISDRODB - A package to standardize, process and analyze global disdrometer data.

|                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Deployment        | [![PyPI](https://badge.fury.io/py/disdrodb.svg?style=flat)](https://pypi.org/project/disdrodb/) [![Conda](https://img.shields.io/conda/vn/conda-forge/disdrodb.svg?logo=conda-forge&logoColor=white&style=flat)](https://anaconda.org/conda-forge/disdrodb)                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Activity          | [![PyPI Downloads](https://img.shields.io/pypi/dm/disdrodb.svg?label=PyPI%20downloads&style=flat)](https://pypi.org/project/disdrodb/) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/disdrodb.svg?label=Conda%20downloads&style=flat)](https://anaconda.org/conda-forge/disdrodb)                                                                                                                                                                                                                                                                                                                                                                               |
| Python Versions   | [![Python Versions](https://img.shields.io/badge/Python-3.10%20%203.11%20%203.12-blue?style=flat)](https://www.python.org/downloads/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Supported Systems | [![Linux](https://img.shields.io/github/actions/workflow/status/ltelab/disdrodb/.github/workflows/tests.yml?label=Linux&style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/tests.yml) [![macOS](https://img.shields.io/github/actions/workflow/status/ltelab/disdrodb/.github/workflows/tests.yml?label=macOS&style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/tests.yml) [![Windows](https://img.shields.io/github/actions/workflow/status/ltelab/disdrodb/.github/workflows/tests_windows.yml?label=Windows&style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/tests_windows.yml)                                                |
| Project Status    | [![Project Status](https://www.repostatus.org/badges/latest/active.svg?style=flat)](https://www.repostatus.org/#active)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Build Status      | [![Tests](https://github.com/ltelab/disdrodb/actions/workflows/tests.yml/badge.svg?style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/tests.yml) [![Lint](https://github.com/ltelab/disdrodb/actions/workflows/lint.yml/badge.svg?style=flat)](https://github.com/ltelab/disdrodb/actions/workflows/lint.yml) [![Docs](https://readthedocs.org/projects/disdrodb/badge/?version=latest&style=flat)](https://disdrodb.readthedocs.io/en/latest/)                                                                                                                                                                                                                    |
| Linting           | [![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat)](https://github.com/astral-sh/ruff) [![Codespell](https://img.shields.io/badge/Codespell-enabled-brightgreen?style=flat)](https://github.com/codespell-project/codespell)                                                                                                                                                                                                                                                     |
| Code Coverage     | [![Coveralls](https://coveralls.io/repos/github/ltelab/disdrodb/badge.svg?branch=main&style=flat)](https://coveralls.io/github/ltelab/disdrodb?branch=main) [![Codecov](https://codecov.io/gh/ltelab/disdrodb/branch/main/graph/badge.svg?style=flat)](https://codecov.io/gh/ltelab/disdrodb)                                                                                                                                                                                                                                                                                                                                                                                     |
| Code Quality      | [![Codefactor](https://www.codefactor.io/repository/github/ltelab/disdrodb/badge?style=flat)](https://www.codefactor.io/repository/github/ltelab/disdrodb) [![Codebeat](https://codebeat.co/badges/14ff831b-f064-4bdd-a2e2-72ffdf28a35a?style=flat)](https://codebeat.co/projects/github-com-ltelab-disdrodb-main) [![Codacy](https://app.codacy.com/project/badge/Grade/d823c50a7ad14268bd347b5aba384623?style=flat)](https://app.codacy.com/gh/ltelab/disdrodb/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Codescene](https://codescene.io/projects/36773/status-badges/code-health?style=flat)](https://codescene.io/projects/36773) |

| License | [![License](https://img.shields.io/github/license/ltelab/disdrodb?style=flat)](https://github.com/ltelab/disdrodb/blob/main/LICENSE) |
| Community | [![Slack](https://img.shields.io/badge/Slack-disdrodb-green.svg?logo=slack&style=flat)](https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA) [![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-green?logo=github&style=flat)](https://github.com/ltelab/disdrodb/discussions) |
| Citation | [![GMD](TOOO_ADD_LINK:status.svg?style=flat)](#) [![DOI](https://zenodo.org/badge/429018433.svg?style=flat)](https://zenodo.org/doi/10.5281/zenodo.7680581) |

[**Slack**](https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA) | [**Documentation**](https://disdrodb.readthedocs.io/en/latest/)

DISDRODB is part of an initial effort to index, collect and homogenize drop size distribution (DSD) data sets across the globe,
as well as to establish a global standard for disdrometers observations data sharing.

DISDRODB standards are being established following FAIR data best practices and Climate & Forecast (CF) conventions, and will facilitate
the preprocessing, analysis and visualization of disdrometer data.

## ℹ️ Software Overview

The software enables you to:

- upload raw data of new disdrometer stations to the DISDRODB Decentralized Data Archive

- download the raw disdrometer data from stations included in the DISDRODB Decentralized Data Archive

- convert raw disdrometer data into a standard NetCDF format (DISDRODB L0 product)

- generate standardized, homogenized, and quality-checked disdrometer measurements (DISDRODB L1 product)

- Compute empirical and model-based drop size distribution parameters and derive geophysical and polarimetric radar variables of interest (DISDRODB L2 product)

Currently, the DISDRODB Working Group is finalizing the development of the L1 and L2 scientific products.
If you have ideas, algorithms, data or expertise to share, or you want to contribute to the future DISDRODB products, do not hesitate to get in touch !!!

Join the [**DISDRODB Slack Workspace**](https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA) to meet the DISDRODB Community !

## 🚀 Quick Start

You're about to create your very own DISDRODB Local Data Archive.

### 📚 Download the DISDRODB Metadata Archive

The DISDRODB Metadata Archive is a collection of metadata files that describe the disdrometer stations included in DISDRODB.

To download the DISDRODB Metadata Archive, navigate to the desired directory and run:

```bash

git clone https://github.com/ltelab/DISDRODB-METADATA.git

```

This command will download a directory called `DISDRODB-METADATA`.

Alternatively, you can download a static version of the DISDRODB Metadata Archive using:

```bash
disdrodb_download_metadata_archive /custom/path/where/to/download/DISDRODB-METADATA
```

### 📚 Define the DISDRODB Configuration File

The disdrodb software needs to know where the local DISDRODB Metadata Archive
is stored on your local machine, as well as where you want to download the raw stations data
as well as where to save the DISDRODB products you will generate.

Within the `disdrodb` package, the argument `metadata_archive_dir` refers to the base directory of your local DISDRODB Metadata Archive, while `data_archive_dir` refers to the base directory of your local DISDRODB Data Archive.

The `metadata_archive_dir` path corresponds to the `DISDRODB` directory within the `DISDRODB-METADATA` archive.
The `data_archive_dir` path corresponds to `DISDRODB` directory of choice where
all DISDRODB products will be saved.

To facilitate the creation of the DISDRODB Configuration File, you can adapt and run in python the following code snippet.
Please note that on Windows these paths must end with `\DISDRODB`, while on macOS/Linux they must end with `/DISDRODB`.

```python
import disdrodb

metadata_archive_dir = "\<path_to>/DISDRODB-METADATA/DISDRODB"
data_archive_dir = "\<path_of_choice_to_the_local_data_archive>/DISDRODB"
disdrodb.define_configs(
    metadata_archive_dir=metadata_archive_dir, data_archive_dir=data_archive_dir
)
```

By running this command, the disdrodb software will write a `.config_disdrodb.yml` file into your home directory (i.e. `~/.config_disdrodb.yml`)
that will be used as default configuration file when running the disdrodb software.

If you **now close your python session and reopen a new one**, if you will run the following code snippet, you
should get the `metadata_archive_dir` and `data_archive_dir` paths you just defined in the DISDRODB Configuration File:

```python
import disdrodb

print("DISDRODB Metadata Archive Directory: ", disdrodb.get_metadata_archive_dir())
print("DISDRODB Data Archive Directory: ", disdrodb.get_data_archive_dir())
```

You can also verify and print the default DISDRODB Metadata Archive and Data Archive directories
by typing the following command in the terminal:

```bash

   disdrodb_data_archive_directory
   disdrodb_metadata_archive_directory

```

### 📥 Download the DISDRODB Raw Data Archive

To download all data stored into the DISDRODB Decentralized Data Archive, you just have to run the following command:

```bash
disdrodb_download_archive
```

If you aim to download data from a specific source (e.g., EPFL), run:

```bash
disdrodb_download_archive --data-sources EPFL
```

Type `disdrodb_download_archive --help` to see further options.

To open the local DISDRODB Data Archive directory, type:

```bash
disdrodb_open_data_archive
```

### 💫 Transform the raw data to standardized netCDF files

If you want to convert all stations raw data into standardized netCDF4 files, run the following command in the terminal:

```bash
disdrodb_run_l0
```

Type `disdrodb_run_l0 --help` to see further options.

### 💫 Generate the DISDRODB L1 and L2 products

To generate the DISDRODB L1 and L2 products, run the following commands in the terminal:

```bash
disdrodb_run_l1 disdrodb_run_l2e disdrodb_run_l2m
```

### 💫 Open analysis-ready DISDRODB products

The disdrodb software `open_dataset` function enable to lazy open all station files of
a DISDRODB product into a `xarray.Dataset` (or `pandas.DataFrame` for the DISDRODB L0A product).

```python
import disdrodb

# Define station arguments

data_source = "EPFL"
campaign_name = "HYMEX_LTE_SOP3"
station_name = "10"

# Open all station files of a specific product

ds = disdrodb.open_dataset(
    product="L0C",
    # Station arguments
    data_source=data_source,
    campaign_name=campaign_name,
    station_name=station_name,
)
ds
```

This allows you to jump directly into analyzing disdrometer data without worrying about processing steps.

### 💫 Explore the DISDRODB Metadata Archive

To explore the DISDRODB Metadata Archive, you can type into the terminal:

```bash
disdrodb_open_metadata_archive
```

If you wish to analyze the DISDRODB Metadata Archive information of all stations,
the `read_metadata_archive` python function returns all stations metadata information into an easy to analyze `pandas.DataFrame`:

```python
import disdrodb

df = disdrodb.read_metadata_archive()
print(df)
```

## 📖 Explore the DISDRODB documentation

With this introduction, we just scratched the surface of the disdrodb software capabilities.
To discover more about the DISDRODB products, the download and processing options, or how to contribute your own data to DISDRODB,
please read the software documentation available at [https://disdrodb.readthedocs.io/en/latest/](https://disdrodb.readthedocs.io/en/latest/).

## 🛠️ Installation

### conda

DISDRODB can be installed via [conda][conda_link] on Linux, Mac, and Windows.
Install the package by typing the following command in the terminal:

```bash
conda install disdrodb
```

In case conda-forge is not set up for your system yet, see the easy to follow instructions on [conda-forge][conda_forge_link].

### pip

DISDRODB can be installed also via [pip][pip_link] on Linux, Mac, and Windows.
On Windows you can install [WinPython][winpy_link] to get Python and pip running.

Then, install the DISDRODB package by typing the following command in the terminal:

```bash
pip install disdrodb
```

To install the latest development version via pip, see the [documentation][dev_install_link].

## 💭 Feedback and Contributing Guidelines

If you aim to contribute your data or discuss the future development of DISDRODB,
we highly recommend to join the [**DISDRODB Slack Workspace**](https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA)

Feel free to also open a [GitHub Issue](https://github.com/ltelab/disdrodb/issues) or a
[GitHub Discussion](https://github.com/ltelab/disdrodb/discussions) specific to your questions or ideas.

## ✍️ Contributors

- [Gionata Ghiggi](https://people.epfl.ch/gionata.ghiggi)
- [Kim Candolfi](https://github.com/KimCandolfi)
- [Régis Longchamp](https://people.epfl.ch/regis.longchamp)
- [Charlotte Gisèle Weil](https://people.epfl.ch/charlotte.weil)
- [Jacopo Grazioli](https://people.epfl.ch/jacopo.grazioli)
- [Alexis Berne](https://people.epfl.ch/alexis.berne?lang=en)

## Citation

You can cite the DISDRODB software by:

> Gionata Ghiggi, Kim Candolfi, Régis Longchamp, Charlotte Weil, Alexis Berne (2023). ltelab/disdrodb Zenodo. https://doi.org/10.5281/zenodo.7680581

If you want to cite a specific version, have a look at the [Zenodo site](https://doi.org/10.5281/zenodo.7680581)

## License

The content of this repository is released under the terms of the [GPL 3.0 license](LICENSE).

[conda_forge_link]: https://github.com/conda-forge/disdrodb-feedstock#installing-disdrodb
[conda_link]: https://docs.conda.io/en/latest/miniconda.html
[dev_install_link]: https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-contributors
[pip_link]: https://pypi.org/project/disdrodb
[winpy_link]: https://winpython.github.io/
