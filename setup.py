from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "{{VERSION_PLACEHOLDER}}"
DESCRIPTION = "This package provides tools to homogenize, process, and analyze global disdrometer data."


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
    # Always use forward '/', even on Windows
    # See https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html#data-files-support
    package_data={
        "disdrodb": [
            "L0/configs/*/*.yaml",
            #'L0/readers/*/*.py',
        ]
    },
    install_requires=[
        "click",
        "h5py",
        "netCDF4",
        "pyarrow",
        "PyYAML",
        "setuptools",
        "xarray",
        "dask",
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
