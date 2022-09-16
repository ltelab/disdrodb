=========================
Installation
=========================

.. warning::
    This document is not complete !

    Currently under development.

    Do not use it now.


We define here two types of installation :

- `Installation for users`_ : for users who want to process data.

- `Installation for contributors`_: for contributors who want to enrich the project (eg. add a new reader).




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





Installation for contributors
================================


The latest disdrodb stable version is available on the Github repository `disdrodb <https://github.com/ltelab/disdrodb>`_.

Clone the repository from github
.........................................

According to the `contributors guidelines <contributors_guidelines>`__, you should first create a fork into your personal github account.

* Install a local copy of the forked repository:

.. code-block:: bash

    git clone https://github.com/<your-account>/disdrodb.git
    cd disdrodb


Install the python developing environment
............................................

You can use either conda or pip : 

* **Conda**


	* Install the dependencies using conda:

	.. code-block:: bash

		conda env create -f environment.yml

	* Activate the disdrodb conda environment

	.. code-block:: bash

		conda activate disdrodb-dev


* **Pip**

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


	* Load dependencies:

	.. code-block:: bash

	   pip install -r requirements.txt



Install pre-commit hooks
..............................

After setting up your development environment, install the git
pre-commit hook by executing the following command in the repository’s
root:

::

   pre-commit install

The pre-commit hooks are scripts executed automatically in every commit
to identify simple code quality issues. When an issue is identified
(the pre-commit script exits with non-zero status), the hook aborts the
commit and prints the error. Currently, DISDRODB only tests that the
code to be committed complies with black’s format style. In case that
the commit is aborted, you only need to run black agains you code.
This can be done by running ``black .`` or
``pre-commit run --all-files``. The latter is recommended since it
indicates if the commit contained any formatting errors (that are
automatically corrected).

IMPORTANT: Periodically update the black version used in the pre-commit
hook by running:

::

   pre-commit autoupdate


