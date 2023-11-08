========================
Software Structure
========================

The current software structure is described below:

| 📁 data/
| 📁 disdrodb/
| ├──  📁 api
|       ├── 📜 checks.py
|       ├── 📜 info.py
|       ├── 📜 io.py
| ├── 📁 metadata
|     ├── 📁 scripts
|         ├── 📜 disdrodb_check_metadata_archive.py
|     ├── 📜 check_metadata.py
|     ├── 📜 info.py
|     ├── 📜 io.py
|     ├── 📜 manipulation.py
|     ├── 📜 standards.py
| ├── 📁 data_transfer
|     ├── 📁 scripts
|         ├── 📜 disdrodb_download_archive.py
|         ├── 📜 disdrodb_upload_archive.py
|         ├── 📜 disdrodb_upload_station.py
|     ├── 📜 download_data.py
|     ├── 📜 upload_data.py
|     ├── 📜 zenodo.py
| ├── 📁 l0
|     ├── 📁 configs
|     	├── 📁 `<sensor_name>`
|     		├── 📜 \*.yml
|     ├── 📁 manuals
|       ├── 📜 \*.pdf
|     ├── 📁 readers
|     	├── 📁 `<data_source>`
|           ├── 📜 \<reader_name>.py
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
|     ├── 📜 issue.py
|     ├── 📜 l0_processing.py
|     ├── 📜 l0a_processing.py
|     ├── 📜 l0b_processing.py
|     ├── 📜 l0b_processing.py
|     ├── 📜 l0b_nc_processing.py
|     ├── 📜 l0_reader.py
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
| 📜 CODE_OF_CONDUCT.md
| 📜 CONTRIBUTING.rst
| 📜 environment.yml
| 📜 LICENSE
| 📜 MANIFEST.in
| 📜 pyproject.toml
| 📜 README.md
| 📜 readthedocs.yml
| 📜 requirements.txt
