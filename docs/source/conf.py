# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import shutil
import pandas as pd
import numpy as np

# sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.join(os.path.abspath("../.."), "disdrodb"))
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------

project = "Disdrodb"
copyright = "LTE - Environmental Remote Sensing Lab - EPFL"
author = "LTE - Environmental Remote Sensing Lab - EPFL"


# Copy tutorial notebook
root_path = os.path.dirname(os.path.dirname(os.getcwd()))

in_path = os.path.join(
    root_path, "disdrodb", "L0", "readers", "reader_preparation.ipynb"
)
out_path = os.path.join(os.getcwd(), "reader_preparation.ipynb")

shutil.copyfile(in_path, out_path)


# Get key metadata from google sheet
doc_id = "1z1bh55BFTwp7u-069PD8NF6r_ZmpCQwr7i78W6RBY_g"
sheet_id = "0"
sheet_url = (
    f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={sheet_id}"
)
df_all = pd.read_csv(sheet_url, header=3)


df_list = np.split(df_all, df_all[df_all.isnull().all(1)].index)

# Remove all CSV from the folder
path_csv_folder = os.path.join(root_path, "docs", "source", "csv")
filelist = [f for f in os.listdir(path_csv_folder) if f.endswith(".csv")]
for f in filelist:
    os.remove(os.path.join(path_csv_folder, f))


# create rst text
metadata_keys_text = """
=========================
Metadata keys
=========================


The following tables are based on `this google sheet <https://docs.google.com/spreadsheets/d/1z1bh55BFTwp7u-069PD8NF6r_ZmpCQwr7i78W6RBY_g>`__. 

"""

for current_df in df_list:
    # Remove empty lines
    df = current_df.dropna(how="all")

    # Reset the index
    df = df.reset_index(drop=True)

    # Get the table name
    df_name = df.iloc[0][0]

    # Remove line containing the table names
    df = df.iloc[1:]

    # Remove * in the table content
    df = df.replace("\*", "", regex=True)

    # save csv file
    path_csv_file = os.path.join(path_csv_folder, f"{df_name}.csv")
    df.to_csv(path_csv_file, index=False)

    # populate rst text
    metadata_keys_text += "\n"
    metadata_keys_text += "\n"
    metadata_keys_text += "\n| "
    metadata_keys_text += f"\n{df_name}"
    line = len(df_name) * "="
    metadata_keys_text += f"\n{line}"
    metadata_keys_text += "\n"
    metadata_keys_text += f"\n.. csv-table:: {df_name}"
    path_csv = os.path.join(os.path.dirname(__file__), "csv", f"{df_name}.csv")
    metadata_keys_text += "\n   :align: left"
    metadata_keys_text += f"\n   :file: {path_csv}"
    metadata_keys_text += "\n   :widths: auto"
    metadata_keys_text += "\n   :header-rows: 1"


# Save rst text as file
metadata_keys_file = open("metadata_keys.rst", "w")
metadata_keys_file.writelines(metadata_keys_text)
metadata_keys_file.close()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]
