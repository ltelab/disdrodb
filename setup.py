from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.0"  # VERSION_PLACEHOLDER
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
            "L0/configs/*/*.yml",
            "L0/readers/*/*.py",
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
    entry_points={
        # <name_to_type_in_terminal>=<module>:<name_of_the_function>
        "console_scripts": [
            # L0A
            "run_disdrodb_l0a_station=disdrodb.l0.scripts.run_disdrodb_l0a_station:run_disdrodb_l0a_station",
            "run_disdrodb_l0a=disdrodb.l0.scripts.run_disdrodb_l0a:run_disdrodb_l0a",
            # L0B
            "run_disdrodb_l0b_station=disdrodb.l0.scripts.run_disdrodb_l0b_station:run_disdrodb_l0b_station",
            "run_disdrodb_l0_station=disdrodb.l0.scripts.run_disdrodb_l0_station:run_disdrodb_l0_station",
            # L0B concatenation
            "run_disdrodb_l0b_concat_station=disdrodb.l0.scripts.run_disdrodb_l0b_concat_station:run_disdrodb_l0b_concat_station",
            "run_disdrodb_l0b_concat=disdrodb.l0.scripts.run_disdrodb_l0b_concat:run_disdrodb_l0b_concat",
            # L0
            "run_disdrodb_l0b=disdrodb.l0.scripts.run_disdrodb_l0b:run_disdrodb_l0b",
            "run_disdrodb_l0=disdrodb.l0.scripts.run_disdrodb_l0:run_disdrodb_l0",
        ]
    },
)
