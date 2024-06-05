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
import shutil
import sys
import inspect
import disdrodb

# sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.join(os.path.abspath("../.."), "disdrodb"))


# -- Project information -----------------------------------------------------

project = "disdrodb"
copyright = "LTE - Environmental Remote Sensing Lab - EPFL"
author = "LTE - Environmental Remote Sensing Lab - EPFL"


# Copy tutorial notebook
root_path = os.path.dirname(os.path.dirname(os.getcwd()))
filenames = ["reader_preparation.ipynb"]
for filename in filenames:
    in_path = os.path.join(root_path, "tutorials", filename)
    out_path = os.path.join(os.getcwd(), "tutorials", filename)
    shutil.copyfile(in_path, out_path)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",
    # "sphinx_design",
    # "sphinx_gallery.gen_gallery",
    # "sphinx.ext.autosectionlabel",
    "sphinx_mdinclude",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "myst_parser",
    "nbsphinx",
]

# Set up mapping for other projects' docs
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}
always_document_param_types = True

# Warn when a reference is not found in docstrings
nitpicky = True
nitpick_ignore = [
    ("py:class", "optional"),
    ("py:class", "array-like"),
    ("py:class", "file-like object"),
    # For traitlets docstrings
    ("py:class", "All"),
    ("py:class", "t.Any"),
    ("py:class", "t.Iterable"),
    ("py:class", "sorted"),
    ("py:class", "dictionary"),
]
nitpick_ignore_regex = [
    ("py:class", r".*[cC]allable"),
]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# For a class, combine class and __init__ docstrings
autoclass_content = "both"

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = "disdrodb"
html_theme_options = {
    "repository_url": "https://github.com/ltelab/disdrodb",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": True,
    # "use_source_button": True,
    "use_issues_button": True,
    # "use_repository_button": True,
    "use_download_button": True,
    # "use_sidenotes": True,
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]


# -- Automatically run apidoc to generate rst from code ----------------------
# https://github.com/readthedocs/readthedocs.org/issues/1139
def run_apidoc(_):
    from sphinx.ext.apidoc import main

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    cur_dir = os.path.abspath(os.path.dirname(__file__))

    module_dir = os.path.join(cur_dir, "..", "..", "disdrodb")
    output_dir = os.path.join(cur_dir, "api")
    exclude = [os.path.join(module_dir, "tests")]
    main(["-f", "-o", output_dir, module_dir, *exclude])


def setup(app):
    app.connect("builder-inited", run_apidoc)


# Function to resolve source code links for `linkcode`
# adapted from NumPy, Pandas implementations
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(disdrodb.__file__))

    if "+" in disdrodb.__version__:
        return f"https://github.com/ltelab/disdrodb/blob/main/disdrodb/{fn}{linespec}"
    else:
        return f"https://github.com/ltelab/disdrodb/blob/" f"v{disdrodb.__version__}/disdrodb/{fn}{linespec}"
