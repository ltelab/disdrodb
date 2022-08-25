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

Disdrodb can be installed into a conda environment as follow :


.. code-block:: bash

	conda install -c conda-forge disdrodb

   

Pip-based Installation
..............................

Disdrodb is available from the Python Packaging Index (PyPI) as follow :


.. code-block:: bash

   pip install disdrodb





Installation for developers
============================


The latest stable version disdrodb is available on Github repository  `disdrodb <https://github.com/ltelab/disdrodb>`_.

According to the `contributors guidelines <contributors_guidelines>`__, you should first create a fork into your personal github account. 

Installation from a local copy of the github repository

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


* Create a virtual environment:


.. code-block:: bash

   python -m venv venv

* activate virtual environment

.. code-block:: bash

   cd venv/Script
   activate

.. warning:: 
   To do : Validate this pocess on others OS than Windows


* load dependencies:

.. code-block:: bash

   pip install -r requirements.txt







