=========================
Installation
=========================


We define here two types of installation :

- `Installation for standard users`_ : for users who want to process data.

- `Installation for contributors`_: for contributors who want to enrich the project (eg. add a new reader).

We recommend users and contributors to use a `Virtual environment`_ to install DISDRODB.


Installation for standard users
==================================

Pip-based installation
..............................

disdrodb is available from the `Python Packaging Index (PyPI) <https://pypi.org/>`__ as follow:


.. code-block:: bash

   pip install disdrodb






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






Install the DISDRODB package dependencies
............................................

You can use either conda or pip : 


* **Pip**

	.. code-block:: bash

	   pip install -r requirements.txt
	


	
* **Conda**

	Create the `disdrodb-dev` conda environment and install the required dependencies:

	.. code-block:: bash

		conda env update -f environment.yml 


To install the project in editable mode : 
	
.. code-block:: bash
		
	pip install -e .












Install pre-commit code quality checking
..............................................

After setting up your development environment, install the git
pre-commit hook by executing the following command in the repository’s
root:

.. code-block:: bash

   pip install pre-commit 
   pre-commit install
   

The pre-commit hooks are scripts executed automatically in every commit
to identify simple code quality issues. When an issue is identified
(the pre-commit script exits with non-zero status), the hook aborts the
commit and prints the error. Currently, DISDRODB only tests that the
code to be committed complies with black’s format style. 

In case that the commit is aborted, you only need to run black agains you code.
This can be done by running ``black .`` or ``pre-commit run --all-files``. The latter is recommended since it
indicates if the commit contained any formatting errors (that are automatically corrected).

.. note::
	To maintain consitency, we use Black version `22.8.0` (as defined into `.pre-commit-config.yaml`). Make sure to stick to version.  





Virtual environment
==================================

While not mandatory, utilizing a virtual environment when installing DISDRODB is recommended. Using a virtual environment for installing packages provides isolation of dependencies, easier package management, easier maintenance, improved security, and improved development workflow.



To set up a virtual environment, follow these steps :

* **With venv :**  

	* Windows: Create a virtual environment with venv:

		.. code-block:: bash

		   python -m venv venv
		   cd venv/Scripts
		   activate
		   

	* Mac/Linux: Create a virtual environment with venv:

		.. code-block:: bash

		   virtualenv -p python3 venv
		   source venv/bin/activate



* **With Conda:**

	* Create the `disdrodb-dev` (or anay other name) conda environment:

		.. code-block:: bash

			conda create --name disdrodb-dev python=3.9 --no-default-packages

	* Activate the disdrodb conda environment:

		.. code-block:: bash

			conda activate disdrodb-dev

		
	* Manually add the path of your local copy of disdrodb to the ``PYTHONPATH`` environment variable. 
	  In Linux operating systems, you could add the following line to your ``.bashrc`` file located in the ``/home/<your_username>`` directory: 
	  
		.. code-block:: bash
		
			export PYTHONPATH="${PYTHONPATH}:/path/to/your/local/repo/of/disdrodb/"


