=================
DISDRODB Readers
=================


DISDRODB supports reading and loading data from many input file formats.
The following subsections describe, first, what a DISDRODB reader is and how it can be defined.
Then, it illustrates multiple methods how a DISDRODB reader can be called (i.e. from terminal or within python) to process raw data into DISDRODB L0 products.

What is a reader
-------------------

A DISDRODB reader is a python function encoding all the required information to convert
raw disdrometer text (or netCDF) data into DISDRODB L0A and/or DISDRODB L0B products.

To be more precise, a reader contains:

1. a glob string specifying the pattern to select all files to be processed within a station directory;

2. the name of the variables present in the raw files (i.e. the file header/columns);

3. some special arguments required to open and read the raw files (i.e the delimiter);

4. an optional ad-hoc function to make the raw data compliant with the DISDRODB standards.

If the raw data are text-based files, the reader will take care of first converting the data
into the DISDRODB L0A dataframe format, and subsequently to reshape the data into the DISDRODB L0B netCDF format.
Instead, if the raw data are netCDFs files, the reader will take care of reformatting the source netCDF into
the DISDRODB L0B netCDF format.

In the DISDRODB metadata of each station:

* the ``reader`` key specifies the DISDRODB reader required to process the raw data.

* the ``raw_data_format`` variable specifies whether the source data is in the form of txt or netcdf files.


Available readers
------------------

In the in the disdrodb software, the readers are organized by data source.
You can have a preliminary look on how the readers looks like by exploring
the `DISDRODB.l0.readers directory <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers>`_

The function ``available_readers`` returns a dictionary with all readers currently available within DISDRODB.
By specifying the ``data_sources`` argument, only the readers for the specified data sources are returned.

.. code-block:: python

    from disdrodb.l0 import available_readers

    available_readers()
    available_readers(data_sources=["EPFL", "GPM"])

The dictionary has the following structure:

.. code-block:: text

    {
        "<DataSource1>": [<ReaderName1>, <ReaderName2>],
        ...
        "<DataSourceN>": [<ReaderNameY>, <ReaderNameZ>],
    }


Reader structure
------------------

A reader is a function defined by the following input arguments:

.. code-block:: python

    def reader(
        raw_dir,
        processed_dir,
        station_name,
        # Processing options
        force=False,
        verbose=False,
        parallel=False,
        debugging_mode=False,
    ):
        pass


* ``raw_dir`` : str - The directory path where all the raw data of a specific campaign/network are stored.

        * The path must have the following structure: ``<...>/DISDRODB/Raw/<data_source>/<campaign_name``.
        * Inside the raw_dir directory, the software expects to find the following structure:

            * ``<raw_dir>/data/<station_name>/<raw_files>``
            * ``<raw_dir>/metadata/<station_name>.yml``


* ``processed_dir`` : str - The desired directory path where to save the DISDRODB L0A and L0B products.

        * The path should have the following structure: ``<...>/DISDRODB/Processed/<data_source>/<campaign_name>``
        * The ``<campaign_name>`` must match with the one specified in the ``raw_dir``.
        * For reader testing purposes, you can define i.e. ``/tmp/DISDRODB/Processed/<data_source>/<campaign_name>``


* ``station_name`` : str - Name of the station to be processed.


* ``force`` : bool [true\| **false** ] - Whether to overwrite existing data.

        *  If ``True``, overwrite existing data into destination directories.
        *  If ``False``, raise an error if there are already data into destination directories.


* ``verbose`` : bool [true\| **false** ] - Whether to print detailed processing information into terminal.


* ``debugging_mode`` : bool [true\| **false** ] -  If ``True``, it reduces the amount of data to process.

        * It processes just 3 raw data files.

* ``parallel`` : bool [true\| **false** ] - Whether to process multiple files simultaneously.

        * If ``parallel=False``, the raw files are processed sequentially.
        * If ``parallel=True``, each file is processed in a separate core.


Inside the reader function, a few components must be customized.


Reader components for raw text files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the input raw data are text files, the reader must defines the following components:

1. The ``glob_patterns`` to search for the raw data files within the ``<raw_dir>/data/<station_name>`` directory.

2. The ``column_names`` list defines the header of the raw text file.

3. The ``reader_kwargs`` dictionary containing all specifications to open the text file into
   a pandas dataframe. The possible key-value arguments are listed `here <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_

4. The ``df_sanitizer_fun(df)`` function takes as input the raw dataframe and apply ad-hoc
   processing to make the dataframe compliant to the DISDRODB L0A standards.
   Typically, this function is used to drop columns not compliant with the expected set of DISDRODB variables
   and to create the DISDRODB expected ``time`` column into UTC datetime format.
   In the output dataframe, each row must correspond to a timestep !

It's important to note that the internal L0A processing already takes care of:

* removing rows with undefined timestep

* removing rows with corrupted values

* sanitize string column with trailing spaces

* dropping rows with duplicated timesteps (keeping only the first occurrence)

In the DISDRODB L0A format, the raw precipitation spectrum, named ``raw_drop_number`` ,
it is expected to be defined as a string with a series of values separated by a delimiter like ``,`` or ``;``.
Therefore, the ``raw_drop_number`` field value is expected to look like ``"000,001,002, ..., 001"``
For example, if the ``raw_drop_number`` looks like the following three cases, you need to preprocess it accordingly
into the ``df_sanitizer_fun``:

* Case 1: ``"000001002 ...001"``. Convert to ``"000,001,002, ..., 001"``.  Example `DELFT reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/NETHERLANDS/DELFT.py>`_
* Case 2: ``"000 001 002 ... 001"``. Convert to ``"000,001,002, ..., 001"``.  Example `CHONGQING reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/CHINA/CHONGQING.py>`_
* Case 3: ``",,,1,2,...,,,"``. Convert to ``"0,0,0,1,2,...,0,0,0"``.  Example reader `SIRTA reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/FRANCE/SIRTA_OTT2.py>`_

Finally, the reader will call the ``run_l0a`` function, by passing to it all the above described arguments.

.. code-block:: python

    run_l0a(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        station_name=station_name,
        # Custom arguments of the reader for L0A processing
        glob_patterns=glob_patterns,
        column_names=column_names,
        reader_kwargs=reader_kwargs,
        df_sanitizer_fun=df_sanitizer_fun,
        # Processing options
        force=force,
        verbose=verbose,
        parallel=parallel,
        debugging_mode=debugging_mode,
    )



Reader components for raw netCDF files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the other hand, if the input raw data are netCDF files, the reader must define the following components:

1. The ``glob_patterns`` to search for the raw netCDF files within the ``<raw_dir>/data/<station_name>`` directory.

2. The ``dict_names`` dictionary mapping the dimension and variables names of the source netCDF to the DISDRODB L0B standards.
   Variables not present the ``dict_names`` are dropped from the dataset.
   Variables specified in ``dict_names`` but missing in the dataset, are added as NaN arrays.
   Here is an example of dict_names:

   .. code-block:: python

       dict_names = {
           # Dimensions
           "timestep": "time",
           "diameter_bin": "diameter_bin_center",
           "velocity_bin": "velocity_bin_center",
           # Variables
           "reflectivity": "reflectivity_32bit",
           "precipitation_spectrum": "raw_drop_number",
       }


3. The ``ds_sanitizer_fun(ds)`` function takes as input the raw netCDF file (in xr.Dataset format) and apply ad-hoc
   processing to make the xr.Dataset compliant to the DISDRODB L0B standards.
   Typically, this function is used to drop xr.Dataset coordinates not compliant with the expected set of DISDRODB coordinates.


Finally, the reader will call the ``run_l0b_from_nc`` function, by passing to it all the above described arguments.

.. code-block:: python

    run_l0b_from_nc(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        station_name=station_name,
        # Custom arguments of the reader
        glob_patterns=glob_patterns,
        dict_names=dict_names,
        ds_sanitizer_fun=ds_sanitizer_fun,
        # Processing options
        force=force,
        verbose=verbose,
        parallel=parallel,
        debugging_mode=debugging_mode,
    )



How to add a new reader
--------------------------

Please refers to the dedicated subsection in
`How to Contribute New Data  <https://disdrodb.readthedocs.io/en/latest/contribute_data.html#implement-the-reader-for-your-data>`_.



Reader preparation tutorial
------------------------------

Please visit the following page to access a read-only tutorial notebook:


.. toctree::
   :maxdepth: 1

   reader_preparation

If you want to run an interactive notebook, you need to run jupyter notebook in your local machine. Proceed as follow :

1. Make sure you have the latest version of the code in your local folder.
See the git clone command in the `Installation for contributors <https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-contributors>`_ section.

2. Enter your project virtual environment or conda environment.
Please, refer to the `Installation for contributors <https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-contributors>`_ section if needed.

3. Navigate to the disdrodb folder.

4. Start the Jupyter Notebook with:

	.. code-block:: bash

		python -m notebook

	or

	.. code-block:: bash

		jupyter notebook

	This will open your default web browser with Jupyter Notebook on the main page.


5. Navigate to ``tutorials`` and double click on the ``reader_preparation.ipynb``.

6. Specify the IPython kernel on which to run the Jupyter Notebook.
To do so, first click on the top ``Kernel`` tab, then click on en ``Change Kernel``, and then select your environment.
If the environment is not available, close the Jupyter Notebook, type the following command and relaunch the notebook:

.. code-block:: bash

    python -m ipykernel install --user --name=<YOUR-ENVIRONMENT-NAME>


7. You can now start using the tutorial notebook.
