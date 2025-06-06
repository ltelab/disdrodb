.. _installation:

=========================
Installation
=========================

This section describes two type of installation:

- `Installation for standard users`_: for processing and analyze disdrometer data.
- `Installation for contributors`_: who want to enrich the project (e.g., adding a new reader, data, ...).

We recommend setting up a virtual environment before installing DISDRODB.


.. _virtual_environment:

Virtual Environment Creation
============================

Although optional, using a virtual environment when installing DISDRODB is recommended.

Virtual environments isolate dependencies, simplify package management, improve maintainability,
enhance security, and streamline your development workflow.

Below are two options for creating a virtual environment,
using `venv <https://docs.python.org/3/library/venv.html>`__ or
`conda <https://docs.conda.io/en/latest/>`__ (recommended).

**With conda:**

* Install `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_, `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ or `anaconda <https://docs.anaconda.com/anaconda/install/>`__ if you haven't already installed.
* Create a new conda environment (e.g., *disdrodb-py311*):

.. code-block:: bash

    conda create --name disdrodb-py311 python=3.11 --no-default-packages

* Activate the environment:

.. code-block:: bash

    conda activate disdrodb-py311

**With venv:**

* On Windows, create and activate a virtual environment:

.. code-block:: bash

    python -m venv disdrodb-pyXXX
    cd disdrodb-pyXXX/Scripts
    activate

* On macOS/Linux, create and activate a virtual environment:

.. code-block:: bash

    python3 -m venv disdrodb-pyXXX
    source disdrodb-pyXXX/bin/activate


.. _installation_standard:

Installation for standard users
==================================

The latest disdrodb stable version is available
on the `Python Packaging Index (PyPI) <https://pypi.org/project/disdrodb/>`__
and on the `conda-forge channel <https://anaconda.org/conda-forge/disdrodb>`__.

Therefore you can either install the package with pip or conda (recommended).
Please install the package in the virtual environment you created before !

**With conda:**

.. code-block:: bash

   conda install -c conda-forge disdrodb


.. note::
   In alternative to conda, if you are looking for a lightweight package manager you could use `micromamba <https://micromamba.readthedocs.io/en/latest/>`__.

**With pip:**

.. code-block:: bash

   pip install disdrodb


.. _installation_contributor:

Installation for contributors
================================

The latest disdrodb version is available on the GitHub repository `disdrodb <https://github.com/ltelab/disdrodb>`__.
You can install the package in editable mode, so that you can modify the code and see the changes immediately.
Here below we provide the steps to install the package in editable mode.

Clone the repository from GitHub
......................................

According to the :ref:`contributors guidelines <contributor_guidelines>`, you should first
`create a fork into your personal GitHub account <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`__.

Then create a local copy of the repository you forked with:

.. code-block:: bash

   git clone https://github.com/<your-account>/disdrodb.git
   cd disdrodb

Create the development environment
......................................

We recommend to create a dedicated conda environment for development purposes.
You can create a conda environment (i.e. with python 3.11) with:

.. code-block:: bash

	conda create --name disdrodb-dev-py311 python=3.11 --no-default-packages
	conda activate disdrodb-dev-py311

Install the package dependencies
............................................

.. code-block:: bash

	conda install --only-deps disdrodb


Install the package in editable mode
................................................

Install the disdrodb package in editable mode by executing the following command in the disdrodb repository's root:

.. code-block:: bash

	pip install -e ".[dev]"


Install code quality checks
..............................................

Install the pre-commit hook by executing the following command in the disdrodb repository's root:

.. code-block:: bash

   pre-commit install


Pre-commit hooks are automated scripts that run during each commit to detect basic code quality issues.
If a hook identifies an issue (signified by the pre-commit script exiting with a non-zero status), it halts the commit process and displays the error messages.

.. note::

	The versions of the software used in the pre-commit hooks is specified in the `.pre-commit-config.yaml <https://github.com/ltelab/disdrodb/blob/main/.pre-commit-config.yaml>`__ file. This file serves as a configuration guide, ensuring that the hooks are executed with the correct versions of each tool, thereby maintaining consistency and reliability in the code quality checks.


Further details about pre-commit hooks can be found in the Contributors Guidelines, specifically in the provided in the :ref:`Code quality control <code_quality_control>` section.


Installation of pyTMatrix
================================

To simulate radar polarimetric variables in the DISDRODB L2E and L2M products, you must install the pyTMatrix package.
The original pyTMatrix can be difficult to install on recent Python versions.

The instructions below describe how to install the LTE-maintained fork of pyTMatrix, which is compatible with modern Python interpreters.

1. Install the GNU `Fortran <https://fortran-lang.org/>`__ Compiler (gfortran) and the `Meson Build system <https://mesonbuild.com/>`__.

.. code-block:: bash

   conda install -c conda-forge gfortran meson

2. Clone the LTE-maintained pyTMatrix repository:

.. code-block:: bash
   git clone ttps://github.com/ltelab/pytmatrix-lte.git

3. Enter the newly cloned ``pytmatrix-lte`` directory and install the package in editable mode:

.. code-block:: bash
   cd pytmatrix-lte
   pip install -e .

4. To confirm that everything was installed correctly, run the pytmatrix built-in test suite. Launch Python and execute:

.. code-block:: python

    from pytmatrix.test import test_tmatrix

    test_tmatrix.run_tests()


.. warning::

   Installing pyTMatrix directly via ``pip install git+https://github.com/ltelab/pytmatrix-lte.git`` does *not* work at this time. We welcome contributions to enable this type of installation !


Installation of Tectonic
=============================

`Tectonic <https://tectonic-typesetting.github.io/en-US/>`__ is a modern typesetting system
that can be used to compile LaTeX documents and create PDF files.
If you want to generate automatic summary tables of rain events or DSD parameters within the disdrodb software,
Tectonic must be installed.

You can install Tectonic using conda:

.. code-block:: bash

   conda install -c conda-forge tectonic


Run DISDRODB on Jupyter Notebooks
==================================

If you want to run disdrodb on a `Jupyter Notebook <https://jupyter.org/>`__,
you have to take care to set up the IPython kernel environment where disdrodb is installed.

For example, if your conda/virtual environment is named ``disdrodb-dev``, run:

.. code-block:: bash

   python -m ipykernel install --user --name=disdrodb-dev

When you will use the Jupyter Notebook, by clicking on ``Kernel`` and then ``Change Kernel``, you will be able to select the ``disdrodb-dev`` kernel.
