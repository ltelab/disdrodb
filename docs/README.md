# Introduction

This documentation is generated by Sphinx.

## Getting started

Install Sphinx as follow  :

```
pip install sphinx
```

## Enrich the documentation

1. Read and agree to the contributing guidelines.

2. Run the following command to empty the *build* folder.

```sh
make clean html
```

3. Update the documentation based on the code with the command :
   
```sh
sphinx-apidoc -f -o source/api .. 
```

4. Write new piece of information into the source//*.rst files.


5. Run the following command to populate the build folder.

```sh
make html
```