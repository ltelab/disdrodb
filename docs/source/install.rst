=========================
Installation
=========================

.. warning::
    This document is not complete !

    Currently under development.

    Do not use it now.


We define here two types of installation :

- `Installation for users`_ : for users who want to process data.

- `Installation for developers`_: for contributors who want to enrich the project (eg. add a new reader).




Installation for users
========================

Conda-based installation
.............................................

disdrodb can be installed into a conda environment as follow :


.. code-block:: bash

	conda install -c conda-forge disdrodb



Pip-based installation
..............................

disdrodb is available from the Python Packaging Index (PyPI) as follow :


.. code-block:: bash

   pip install disdrodb





Installation for developers
============================


The latest disdrodb stable version is available on the Github repository `disdrodb <https://github.com/ltelab/disdrodb>`_.

According to the `contributors guidelines <contributors_guidelines>`__, you should first create a fork into your personal github account.

* Install a local copy of the forked repository:

.. code-block:: bash

    git clone https://github.com/<your-account>/disdrodb.git
    cd disdrodb


Conda
..............................

* Install the dependencies using conda:

.. code-block:: bash

	conda env create -f environment.yml

* Activate the disdrodb conda environment

.. code-block:: bash

	conda activate disdrodb


Pip
..............................

* Load dependencies:

.. code-block:: bash

   pip install -r requirements.txt

* (Optional) We recommend you install in a virtual environment, for example with venv:

	* Windows: Create a virtual environment with venv:

		.. code-block:: bash

		   python -m venv venv
		   cd venv/Script
		   activate

	* Mac/Linux: Create a virtual environment with venv:

		.. code-block:: bash

		   virtualenv -p python3 venv
		   source venv/bin/activate



