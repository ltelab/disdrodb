.. _disdrodb_readers:

=================
Readers
=================


DISDRODB supports reading and loading data from many input file formats.

The following subsections describe, first, what a DISDRODB reader is and how it can be defined.

Then, it illustrates multiple methods how a DISDRODB reader can be called (i.e. from terminal or within python) to process raw data into DISDRODB L0 products.

If you are looking for the ``DISDRODB Reader Implementation Tutorial`` click the link here below:

.. toctree::
   :maxdepth: 1

   tutorials/reader_preparation


What is a reader
-------------------

A DISDRODB reader is a python function responsible for reading one raw data file and converting it into a DISDRODB-compliant object.

Depending on the raw data file format, the reader will produce either an L0A ``pandas.DataFrame`` or an L0B ``xarray.Dataset``.
When it ingest a raw text file, the reader has to output a DISDRODB L0A ``pandas.Dataframe``,
while when it ingest a raw netCDF file, the reader has to output a DISDRODB L0B ``xarray.Dataset``.

For raw text files, the reader function:

1. defines the appropriate options (i.e delimiter, header row, and encoding) required to read the raw text file into a ``pandas.Dataframe``;

2. loads the the raw text file into a ``pandas.Dataframe``, assigning the correct variable names to the columns of the dataframe;

3. adapts the ``pandas.Dataframe`` to the DISDRODB L0A standards, which involves for example dropping columns not part of the DISDRODB variables set and
   ensuring the presence of a UTC ``time`` column in datetime type format;

4. outputs the ``pandas.Dataframe`` in the DISDRODB L0A format.


In the case of raw netCDF files, the reader function:

1. opens the netCDF file into an ``xarray.Dataset`` object;

2. rename the dataset variables to the expected DISDRODB variables set;

3. adapts the ``xarray.Dataset`` to the DISDRODB L0B standards, which involves for example dropping
   variables not included in the expected set of DISDRODB variables.

4. outputs the ``xarray.Dataset`` in the DISDRODB L0B format.


In both cases, the reader encapsulates both the parsing logic for a single-file format and the cleanup rules needed
to bring raw measurements into the standardized DISDRODB format.


In the DISDRODB metadata of each station:

* the ``reader`` reference points the disdrodb software to the reader required to process the station raw data.

* the ``raw_data_format`` variable specifies whether the source raw data is in the form of ``txt`` or ``netcdf``  files.

* the ``raw_data_glob_pattern`` defines which raw data files in the ``DISDRODB/RAW/<DATA_SOURCE>/<CAMPAIGN_NAME>/<STATION_NAME>/data`` directory will be ingested
  in the DISDRODB L0 processing chain.


Available readers
------------------

In the disdrodb software, the readers are organized by sensor name and data source.
You can have a look on how the readers looks like by exploring
the `DISDRODB.l0.readers directory <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers>`_.

You can open the local disdrodb software readers directory typing in the terminal:

.. code:: bash

    disdrodb_open_readers_directory


In python, the function ``available_readers`` returns a list with all readers available for a given sensor.
By specifying the optional ``data_sources`` argument, only the readers references for the specified data sources are returned.

.. code-block:: python

    from disdrodb.l0 import available_readers

    sensor_name = "OTT_Parsivel"
    available_readers(sensor_name)
    available_readers(sensor_name=sensor_name, data_sources=["EPFL", "GPM"])


When you know the reader reference, you can easily retrieve the reader function by using the ``get_reader`` function:

.. code-block:: python

    import disdrodb

    reader = disdrodb.get_reader(reader_reference="EPFL/LOCARNO_2018", sensor_name="OTT_Parsivel")


Alternatively, if you are looking for the reader of a specific station, you can use the ``get_station_reader`` function:

.. code-block:: python

    import disdrodb

    reader = disdrodb.get_station_reader(
        data_source="EPFL",
        campaign_name="LOCARNO_2018",
        station_name="60",
    )



.. _reader_structure:

Reader structure
------------------

In the following two subsections we detail the structure of the disdrodb readers
for ingesting raw text files and raw netCDF files.


Reader for raw text files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reader function for ingesting raw text files is typically structured as follow:

.. code-block:: python

    def reader(filepath, logger=None):
        """Reader."""
        ##-------------------------------------------------------------.
        #### - Define the column names
        column_names = []  # [ADD THE COLUMN NAMES LIST HERE]

        ##-------------------------------------------------------------.
        #### Define reader options
        reader_kwargs = {}
        # - Define delimiter
        reader_kwargs["delimiter"] = ","  # [THIS MIGHT BE CUSTOMIZED]
        # - Skip a specific number of rows
        reader_kwargs["skiprows"] = None  # [THIS MIGHT BE CUSTOMIZED]
        # - Avoid first column to become df index
        reader_kwargs["index_col"] = False

        # [...]

        ##-------------------------------------------------------------.
        #### Read the data
        df = read_raw_text_file(
            filepath=filepath,
            column_names=column_names,
            reader_kwargs=reader_kwargs,
            logger=logger,
        )

        ##-------------------------------------------------------------.
        #### Adapt the dataframe to adhere to DISDRODB L0 standards
        # [ADD YOUR CUSTOM CODE HERE]

        return df


In the reader function:

1. The ``column_names`` list defines the header of the raw text file.

2. The ``reader_kwargs`` dictionary contains all specifications to open the text file into
   a ``pandas.DataFrame``. The possible key-value arguments are listed `here <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_

3. The last part of the reader function code take care of apply ad-hoc
   processing to make the ``pandas.DataFrame`` compliant with the DISDRODB L0A standards.
   Typically, the reader include code to drop columns not compliant with the expected set of DISDRODB variables
   and to create a UTC ``time`` column into datetime type format.
   In the returned ``pandas.DataFrame``, each row must correspond to a timestep !

In the DISDRODB L0A format, the raw precipitation spectrum, named ``raw_drop_number`` ,
it is expected to be defined as a string with a series of values separated by a delimiter like ``,`` or ``;``.
Therefore, the ``raw_drop_number`` field value is expected to look like ``"000,001,002, ..., 001"``.

For example, if the ``raw_drop_number`` strings looks like one of the following three cases,
in the last part of the reader function you need to take care of processing the ``raw_drop_number`` column
and convert it to the expected format:

* Case 1: ``"000001002 ...001"``. Convert to ``"000,001,002, ..., 001"``.  See `DELFT reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/NETHERLANDS/DELFT.py>`_.
* Case 2: ``"000 001 002 ... 001"``. Convert to ``"000,001,002, ..., 001"``.  See `CHONGQING reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/CHINA/CHONGQING.py>`_.
* Case 3: ``",,,1,2,...,,,"``. Convert to ``"0,0,0,1,2,...,0,0,0"``.  See `SIRTA reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/FRANCE/SIRTA_OTT2.py>`_.

When a text reader is invoked by the DISDRODB L0A processing chain, the disdrodb software
automatically applies the following cleaning steps to the ``pandas.DataFrame``:

* removes any rows with undefined timesteps,

* filters out rows that contain corrupted values,

* trims trailing spaces from all string-type columns,

* drop duplicated timesteps, keeping only the first occurrence of each.

Because these checks are already applied downstream, you don't need to implement them yourself in the reader function.

If you want to manually apply the DISDRODB L0A processing chain cleaning steps,
you can simply pass the ``pandas.DataFrame`` returned by the reader to the ``sanitize_df`` function:

.. code-block:: python

    import disdrodb
    from disdrodb.l0.l0a_processing import sanitize_df

    filepath = "path/to/your/raw/text/file.txt"  # [ADAPT TO YOUR FILEPATH]
    sensor_name = sensor_name = "OTT_Parsivel"  # [ADAPT TO YOUR SENSOR_NAME]
    reader_reference = "EPFL/LOCARNO_2018"  # [ADAPT TO YOUR READER]
    reader = disdrodb.get_reader(reader_reference=reader_reference, sensor_name=sensor_name)
    df = reader(filepath)
    df = sanitize_df(df)


The raw text files reader template is available `here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/template_reader_raw_text_data.py>`_.


Reader for raw netCDF files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reader function for ingesting raw netCDF files is typically structured as follow:

.. code-block:: python

    def reader(filepath, logger=None):
        """Reader."""
        ##---------------------------------------------------------------------.
        #### Open the netCDF
        ds = open_raw_netcdf_file(filepath=filepath, logger=logger)

        ##---------------------------------------------------------------------.
        #### Adapt the dataframe to adhere to DISDRODB L0 standards
        # Define dictionary mapping dataset variables and coordinates to keep (and rename)
        # - If the platform is moving, keep longitude, latitude and altitude
        # - If the platform is fixed, remove longitude, latitude and altitude coordinates
        #   --> The geolocation information must be specified in the station metadata !
        dict_names = {
            # Dimensions
            "<timestep>": "time",  # [TO ADAPT]
            "<raw_dataset_diameter_dimension>": "diameter_bin_center",  # [TO ADAPT]
            "<raw_dataset_velocity_dimension>": "velocity_bin_center",  # [TO ADAPT]
            # Variables
            # - Add here other variables accepted by DISDRODB L0 standards
            "<precipitation_spectrum>": "raw_drop_number",  # [TO ADAPT]
        }

        # Rename dataset variables and columns and infill missing variables
        sensor_name = "Thies_LPM"  # [SPECIFY HERE THE SENSOR FOR WHICH THE READER IS DESIGNED]
        ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name=sensor_name)

        # Replace occureence of NaN flags with np.nan
        # - Define a dictionary specifying the value(s) of NaN flags for each variable
        # - The code here below is just an example that requires to be adapted !
        # - This step might not be required with your data !
        dict_nan_flags = {"<raw_drop_number>": [-9999, -999]}
        ds = replace_custom_nan_flags(ds, dict_nan_flags=dict_nan_flags, logger=logger)

        # [ADD ADDITIONAL REQUIRED CUSTOM CODE HERE]

        return ds


In the reader function:

1. The ``dict_names`` dictionary mapping the dimension and variables names of the source netCDF to the DISDRODB L0B standards.
   Variables not present the ``dict_names`` are dropped from the dataset.
   Variables specified in ``dict_names`` but missing in the dataset, are added as NaN arrays.

2. The last part of the reader function code takes care of apply ad-hoc
   processing to make the ``xarray.Dataset`` compliant with the DISDRODB L0B standards.


When a netCDF reader is invoked by the DISDRODB L0B processing chain, the disdrodb software
automatically applies the following cleaning steps to the ``xarray.Dataset``:

* replace classical nan flags values with ``np.nan`` values,

* replace invalid value to ``np.nan``,

* set values outside the data range to ``np.nan``.

Because these checks are already applied downstream, you don't need to implement them yourself in the reader function.

If you want to manually apply the DISDRODB L0B processing chain cleaning steps,
you can simply pass the ``xarray.Dataset`` returned by the reader to the ``sanitize_ds`` function:

.. code-block:: python

    import disdrodb
    from disdrodb.l0.l0b_nc_processing import sanitize_ds

    filepath = "path/to/your/raw/text/file.nc"  # [ADAPT TO YOUR FILEPATH]
    sensor_name = sensor_name = "OTT_Parsivel"  # [ADAPT TO YOUR SENSOR_NAME]
    reader_reference = "EPFL/LOCARNO_2018"  # [ADAPT TO YOUR READER]
    reader = disdrodb.get_reader(reader_reference=reader_reference, sensor_name=sensor_name)
    ds = reader(filepath)
    ds = sanitize_ds(ds)


The raw netCDF files reader template is available `here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/template_reader_raw_netcdf_data.py>`_.



How to develop a new reader
-----------------------------

The `Reader Implementation Tutorial <https://disdrodb.readthedocs.io/en/latest/tutorials/reader_preparation.html>`_ subsection provides read-only access to the DISDRODB Reader Implementation Tutorial.
The original Jupyter Notebook tutorial is available in the disdrodb ``/tutorials`` repository (`here <https://github.com/ltelab/disdrodb/blob/main/tutorials/reader_preparation.ipynb>`_) and can be adapted
to implement new readers .

Please refers to the :ref:`Step 8: Implement the reader <step8>` subsection of the
`How to Contribute New Data <https://disdrodb.readthedocs.io/en/latest/contribute_data.html>`_ section of the documentation for further detailed information.
