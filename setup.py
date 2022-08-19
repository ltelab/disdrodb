from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.1"
DESCRIPTION = "This package provides an API to query, filter and visualize disdrometer data."



# Setting up
setup(
    name="disdrodb",
    version=VERSION,
    author="Gionata Ghiggi",
    author_email="",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "click",
        "dask",
        "h5py",
        "netCDF4",
        "numpy",
        "pandas",
        "pyarrow",
        "PyYAML",
        "setuptools",
        "xarray"
        ],
    keywords=["python", "disdrometer"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
