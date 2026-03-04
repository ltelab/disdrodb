.. _disdrodb_readers:

========
Readers
========

DISDRODB supports reading and loading data from many input file formats.

This guide describes:

1. What a DISDRODB reader is and how it is defined
2. How to call a DISDRODB reader (from terminal or Python) to process raw data into DISDRODB L0 products
3. How to develop new readers for custom data formats

For a hands-on tutorial on implementing a new reader, see:

.. toctree::
   :maxdepth: 1

   tutorials/reader_preparation


What is a Reader?
-------------------

A DISDRODB reader is a Python function that reads one raw data file and converts it into a DISDRODB-compliant format.

**Output Format Depends on Input Type:**

Depending on the raw data file format, the reader produces either:

- **L0A**: A ``pandas.DataFrame`` (for raw text files)
- **L0B**: An ``xarray.Dataset`` (for raw NetCDF files)

**Reader for Raw Text Files**

For raw text files, the reader function:

1. Defines the appropriate options (delimiter, header row, encoding) to read the raw text file into a ``pandas.DataFrame``

2. Loads the raw text file into a ``pandas.DataFrame``, assigning correct column names

3. Adapts the ``pandas.DataFrame`` to DISDRODB L0A standards (drops non-DISDRODB columns, ensures a UTC ``time`` column in datetime format)

4. Returns the ``pandas.DataFrame`` in DISDRODB L0A format


**Reader for Raw NetCDF Files**

In the case of raw NetCDF files, the reader function:

1. Opens the file into an ``xarray.Dataset``

2. Renames dataset variables to match DISDRODB conventions

3. Adapts the ``xarray.Dataset`` to DISDRODB L0B standards (drops variables not in the expected set)

4. Returns the ``xarray.Dataset`` in DISDRODB L0B format


**Purpose of Readers**

In both cases, the reader encapsulates file parsing logic and cleanup rules to standardize
raw measurements into the DISDRODB format.

**Reader Configuration in Station Metadata**

In the DISDRODB metadata for each station:

* The ``reader`` field references the reader function required to process the station's raw data

* The ``raw_data_format`` variable specifies whether the source data are text (``txt``) or NetCDF (``netcdf``) files

* The ``raw_data_glob_pattern`` defines which raw data files in the ``DISDRODB/RAW/<DATA_SOURCE>/<CAMPAIGN_NAME>/<STATION_NAME>/data`` directory
  will be ingested during the DISDRODB L0 processing chain


Available Readers
------------------

In the disdrodb software, readers are organized by sensor name and data source.
You can explore existing readers in the `DISDRODB.l0.readers directory <https://github.com/ltelab/disdrodb/tree/main/disdrodb/l0/readers>`_.

**Open Readers Directory**

To open the local disdrodb readers directory, use the terminal command:

.. code:: bash

    disdrodb_open_readers_directory


**List Available Readers in Python**

The ``available_readers`` function returns a list of all readers available for a given sensor.
By specifying the optional ``data_sources`` argument, you can filter readers for specific data sources:

.. code-block:: python

    from disdrodb.l0 import available_readers

    sensor_name = "PARSIVEL"
    available_readers(sensor_name)
    available_readers(sensor_name=sensor_name, data_sources=["EPFL", "NASA"])


**Get a Specific Reader**

Once you know the reader reference, you can retrieve the reader function using ``get_reader``:

.. code-block:: python

    import disdrodb

    reader = disdrodb.get_reader(reader_reference="EPFL/LOCARNO_2018", sensor_name="PARSIVEL")


**Get Reader for a Specific Station**

Alternatively, if you want the reader for a specific station, use the ``get_station_reader`` function:

.. code-block:: python

    import disdrodb

    reader = disdrodb.get_station_reader(
        data_source="EPFL",
        campaign_name="LOCARNO_2018",
        station_name="60",
    )



Reader Structure
------------------

The following subsections detail the structure of DISDRODB readers
for ingesting raw text files and raw NetCDF files.


Reader for Raw Text Files
~~~~~~~~~~~~~~~~~~~~~~~~~

The reader function for ingesting raw text files is typically structured as follows:

.. code-block:: python

    def reader(filepath, logger=None):
        """Reader."""
        ##-------------------------------------------------------------.
        #### Define the column names
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


**Reader Function Components:**

1. The ``column_names`` list defines the header (column names) of the raw text file

2. The ``reader_kwargs`` dictionary contains all specifications for opening the text file into
   a ``pandas.DataFrame``. The possible key-value arguments are listed in `pandas.read_csv <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_

3. The last part of the reader function applies ad-hoc processing to make the ``pandas.DataFrame``
   compliant with DISDRODB L0A standards. Typically, this includes:

   - Dropping columns not compliant with the expected set of DISDRODB variables
   - Creating a UTC ``time`` column in datetime format
   - Ensuring each row corresponds to one timestep

**Raw Drop Number Format**

In the DISDRODB L0A format, the raw precipitation spectrum (``raw_drop_number``) must be
defined as a string with values separated by a delimiter such as ``,`` or ``;``.
The ``raw_drop_number`` field value should look like ``"000,001,002,...,001"``.

**Examples of Raw Drop Number Conversion:**

If the ``raw_drop_number`` strings in your raw data look like any of the following cases,
you need to convert them to the expected format in the reader function:

* Case 1: ``"000001002 ...001"``. Convert to ``"000,001,002, ..., 001"``.  See `DELFT reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/NETHERLANDS/DELFT.py>`_.
* Case 2: ``"000 001 002 ... 001"``. Convert to ``"000,001,002, ..., 001"``.  See `CHONGQING reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/CHINA/CHONGQING.py>`_.
* Case 3: ``",,,1,2,...,,,"``. Convert to ``"0,0,0,1,2,...,0,0,0"``.  See `SIRTA reader here <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/FRANCE/SIRTA_OTT2.py>`_.

**Automatic Data Cleaning**

When a text reader is invoked by the DISDRODB L0A processing chain, the software
automatically applies the following cleaning steps to the ``pandas.DataFrame``:

* Removes rows with undefined timesteps

* Filters out rows containing corrupted values

* Trims trailing spaces from all string-type columns

* Drops duplicated timesteps, keeping only the first occurrence

Because these checks are applied automatically downstream, you don't need to implement them in the reader function.

**Manual Application of Cleaning Steps**

If you want to manually apply the DISDRODB L0A processing chain cleaning steps,
you can simply pass the ``pandas.DataFrame`` returned by the reader to the ``sanitize_df`` function:

.. code-block:: python

    import disdrodb
    from disdrodb.l0.l0a_processing import sanitize_df

    filepath = "path/to/your/raw/text/file.txt"  # [ADAPT TO YOUR FILEPATH]
    sensor_name = "PARSIVEL"  # [ADAPT TO YOUR SENSOR_NAME]
    reader_reference = "EPFL/LOCARNO_2018"  # [ADAPT TO YOUR READER]
    reader = disdrodb.get_reader(reader_reference=reader_reference, sensor_name=sensor_name)
    df = reader(filepath)
    df = sanitize_df(df)


**Reader Template**

A reader template for raw text files is available at
`https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/template_reader_raw_text_data.py <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/template_reader_raw_text_data.py>`_.


Reader for Raw NetCDF Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reader function for ingesting raw NetCDF files is typically structured as follows:

.. code-block:: python

    def reader(filepath, logger=None):
        """Reader."""
        ##---------------------------------------------------------------------.
        #### Open the netCDF file
        ds = open_raw_netcdf_file(filepath=filepath, logger=logger)

        ##---------------------------------------------------------------------.
        #### Adapt the dataset to DISDRODB L0 standards
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
        sensor_name = "LPM"  # [SPECIFY HERE THE SENSOR FOR WHICH THE READER IS DESIGNED]
        ds = standardize_raw_dataset(ds=ds, dict_names=dict_names, sensor_name=sensor_name)

        # Replace occurrence of NaN flags with np.nan
        # - Define a dictionary specifying the value(s) of NaN flags for each variable
        # - The code here below is just an example that requires to be adapted !
        # - This step might not be required with your data !
        dict_nan_flags = {"<raw_drop_number>": [-9999, -999]}
        ds = replace_custom_nan_flags(ds, dict_nan_flags=dict_nan_flags, logger=logger)

        # [ADD ADDITIONAL REQUIRED CUSTOM CODE HERE]

        return ds


**Reader Function Components:**

1. The ``dict_names`` dictionary maps the dimension and variable names of the source NetCDF to DISDRODB L0B standards.

   - Variables not present in ``dict_names`` are dropped from the dataset
   - Variables specified in ``dict_names`` but missing in the dataset are added as NaN arrays

2. The last part of the reader function applies ad-hoc processing to make the ``xarray.Dataset``
   compliant with DISDRODB L0B standards.

**Automatic Data Cleaning**
When a NetCDF reader is invoked by the DISDRODB L0B processing chain, the software
automatically applies the following cleaning steps to the ``xarray.Dataset``:

* Replaces classical NaN flag values with ``np.nan``

* Replaces invalid values with ``np.nan``

* Sets values outside the valid data range to ``np.nan``

Because these checks are applied automatically downstream, you don't need to implement them in the reader function.

**Manual Application of Cleaning Steps**

If you want to manually apply the DISDRODB L0B processing chain cleaning steps,
you can simply pass the ``xarray.Dataset`` returned by the reader to the ``sanitize_ds`` function:

.. code-block:: python

    import disdrodb
    from disdrodb.l0.l0b_nc_processing import sanitize_ds

    filepath = "path/to/your/raw/text/file.nc"  # [ADAPT TO YOUR FILEPATH]
    sensor_name = "PARSIVEL"  # [ADAPT TO YOUR SENSOR_NAME]
    reader_reference = "EPFL/LOCARNO_2018"  # [ADAPT TO YOUR READER]
    reader = disdrodb.get_reader(reader_reference=reader_reference, sensor_name=sensor_name)
    ds = reader(filepath)
    ds = sanitize_ds(ds)


**Reader Template**

A reader template for raw NetCDF files is available at `https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/template_reader_raw_netcdf_data.py <https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/readers/template_reader_raw_netcdf_data.py>`_.



How to Develop a New Reader
-----------------------------

The `Reader Implementation Tutorial <https://disdrodb.readthedocs.io/en/latest/tutorials/reader_preparation.html>`_ provides a step-by-step guide to implementing a new reader.
The `original Jupyter Notebook tutorial <https://github.com/ltelab/disdrodb/blob/main/tutorials/reader_preparation.ipynb>`_ is available in the disdrodb ``/tutorials`` directory and can be adapted
for implementing new readers.

For detailed information, see the :ref:`Step 8: Implement the reader <step8>` subsection of the
`How to Contribute New Data <https://disdrodb.readthedocs.io/en/latest/contribute_data.html>`_ documentation.
