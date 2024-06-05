========================
Software Structure
========================

The current software structure is described below:

| 📁 data/
| 📁 disdrodb/
| ├──  📁 api
|     ├── 📁 scripts
|         ├── 📜 disdrodb_initialize_station.py.py
|       ├── 📜 checks.py
|       ├── 📜 create_directories.py
|       ├── 📜 info.py
|       ├── 📜 io.py
|       ├── 📜 path.py
| ├── 📁 metadata
|     ├── 📁 scripts
|         ├── 📜 disdrodb_check_metadata_archive.py
|     ├── 📜 checks.py
|     ├── 📜 info.py
|     ├── 📜 search.py
|     ├── 📜 reader.py
|     ├── 📜 writer.py
|     ├── 📜 manipulation.py
|     ├── 📜 standards.py
| ├── 📁 issue
|     ├── 📜 checks.py
|     ├── 📜 reader.py
|     ├── 📜 writer.py
| ├── 📁 data_transfer
|     ├── 📁 scripts
|         ├── 📜 disdrodb_download_archive.py
|         ├── 📜 disdrodb_download_station.py
|         ├── 📜 disdrodb_upload_archive.py
|         ├── 📜 disdrodb_upload_station.py
|     ├── 📜 download_data.py
|     ├── 📜 upload_data.py
|     ├── 📜 zenodo.py
| ├── 📁 l0
|     ├── 📁 configs
|     	├── 📁 *<sensor_name>*
|     		├── 📜 \*.yml
|     ├── 📁 manuals
|       ├── 📜 \*.pdf
|     ├── 📁 readers
|     	├── 📁 *<DATA_SOURCE>*
|           ├── 📜 \<READER_NAME>.py
|     ├── 📁 scripts
|         ├── 📜 disdrodb_run_l0_station.py
|         ├── 📜 disdrodb_run_l0
|         ├── 📜 disdrodb_run_l0a.py
|         ├── 📜 disdrodb_run_l0a_station.py
|         ├── 📜 disdrodb_run_l0b.py
|         ├── 📜 disdrodb_run_l0b_station.py
|         ├── 📜 disdrodb_run_l0b_concat.py
|         ├── 📜 disdrodb_run_l0b_concat_station.py
|     ├── 📜 check_configs.py
|     ├── 📜 check_standards.py
|     ├── 📜 io.py
|     ├── 📜 l0_processing.py
|     ├── 📜 l0a_processing.py
|     ├── 📜 l0b_processing.py
|     ├── 📜 l0b_processing.py
|     ├── 📜 l0b_nc_processing.py
|     ├── 📜 l0_reader.py
|     ├── 📜 routines.py
|     ├── 📜 standards.py
|     ├── 📜 summary.py
|     ├── 📜 template_tools.py
| ├── 📁 l1/
| ├── 📁 l2/
| ├── 📁 tests/
|   ├── 📜 \*.py
| ├── 📁 api/
| ├── 📁 utils/
|   ├── 📜 logger.py
|   ├── 📜 scripts.py
|   ├── 📜 netcdf.py
|   ├── 📜 yaml.py
| 📁 docs/
| 📁 tutorials
| 📜 .gitignore
| 📜 .pre-commit-config.yaml
| 📜 .readthedocs.yml
| 📜 pyproject.toml
| 📜 MANIFEST.in
| 📜 CODE_OF_CONDUCT.md
| 📜 CONTRIBUTING.rst
| 📜 LICENSE
| 📜 README.md
