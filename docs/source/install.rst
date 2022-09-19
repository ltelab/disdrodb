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

Pip-based installation
..............................

disdrodb is available from the `Python Packaging Index (PyPI) <https://pypi.org/>`__ as follow:


.. code-block:: bash

   pip install disdrodb


Conda-based installation is not yet available !!!
.............................................

In future, it will be possible to install disdrodb using the following command:


.. code-block:: bash

	conda install -c conda-forge disdrodb




Installation for contributors
================================


The latest disdrodb stable version is available on the Github repository `disdrodb <https://github.com/ltelab/disdrodb>`_.

Clone the repository from github
.........................................

According to the `contributors guidelines <contributors_guidelines>`__, you should first create a fork into your personal GitHub account.

* Install a local copy of the forked repository:

.. code-block:: bash

    git clone https://github.com/<your-account>/disdrodb.git
    cd disdrodb


Install the python developing environment
............................................

You can use either conda or pip : 

* **Conda (Option 1)**


	* Create the `disdrodb-dev` conda environment:

	.. code-block:: bash
		conda create --name disdrodb_dev python=3.9 --no-default-packages
		
		conda env create -f environment.yml

	* Activate the disdrodb conda environment:

	.. code-block:: bash

		conda activate disdrodb-dev
		
	* Go inside your disdrodb local repository:	
	
	.. code-block:: bash

		cd /path/to/your/local/disdro/repository
	
	
	* Install the required dependencies:
	
	.. code-block:: bash

		pip install -r requirements.txt
	
	.. warning::
		Note: In future, when the conda disdrodb feedstock installation will be set up, the following code should be used instead: 
	
		.. code-block:: bash
	
	     		conda install --only-deps disdrodb
		
	* Install disdrodb 
	
	.. code-block:: bash
	
		pip install -e .
	
		
* **Conda (Option 2)**


	* Create the `disdrodb-dev` conda environment and install the required dependencies:

	.. code-block:: bash

		conda env create -f environment.yml 
	
	.. warning::
		
		Note: This command takes quite some time at the moment ... 

	* Activate the disdrodb conda environment

	.. code-block:: bash

		conda activate disdrodb-dev
		
	* Manually add the path of your local copy of disdrodb to the ``PYTHONPATH`` environment variable. 
	  In Linux operating systems, you could add the following line to your ``.bashrc`` file located in the ``/home/<your_username>`` directory: 
	  
	  .. code-block:: bash
	  	export PYTHONPATH="${PYTHONPATH}:/path/to/your/local/repo/of/disdrodb/"


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
   pip install pre-commit 
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


