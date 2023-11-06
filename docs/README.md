# Introduction

DISDRODB’s documentation is built using Sphinx. All documentation lives in the `docs/` directory of the project repository.

## Getting started to generate the documentation locally

### Set the python environment

Sphinx should be installed within your python environment. Use the virtual environment of the disdrodb project. More information about the virtual environment can be found in the [installation guide](https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-standard-users).



### Generate the documentation

Run the following command to generate the doc :

```
sphinx-build -b html source build
```


The output of the previous command should be checked for warnings and errors. In case of any changes made to the code such as adding new classes or
functions, it is necessary to regenerate the disdrodb documentation files before running the command mentioned above :

    sphinx-apidoc -f -o source/api .. ../setup.py
