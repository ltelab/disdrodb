# DISDRO API - An API to query didsdrometer data.

The code in this repository provides an API to query, filter and visualize disdrometer data.
Documentation is available at XXXXX

ATTENTION: The code is subject to changes in the coming  months.

The folder `tutorials` (will) provide jupyter notebooks describing various features of DISDRODB API.

- How to contribute a custom parser to DISDRODB [[`parser_development.ipynb`]]
- Downloading current DISDRODB dataset [[`download.ipynb`]]
- Read and filter DISDRODB [[`read_and_filter.ipynb`]]
- Exploratory Data Analysis of DISDRODB [[`eda.ipynb`]]

[`download.ipynb`]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-weather/blob/outputs/tutorials/spherical_grids.ipynb
[`eda.ipynb`]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-weather/blob/outputs/tutorials/interpolation_pooling.ipynb

The folder `templates` provide parser template to contribute your data to DISDRODB.

## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
   
   ```sh
   git clone git@github.com:ltelab/disdrodb.git
   cd disdrodb
   ```

2. Install the dependencies using conda:
   
   ```sh
   conda env create -f environment.yml
   ```

3. Activate the mascdb conda environment 
   
   ```sh
   conda activate disdrodb
   ```

4. Just for info... to update the environment.yml: 
   
   ```sh
   conda env export > environment.yml
   ```

## References

- [Slides](https://docs.google.com/presentation/d/1X_MJIXGMGpmXeZh-W6PIgzeO1_0p_ZOAh-ikKNUFQcE/edit?usp=sharing)
- [Manuscript](https://XXXX)
- [Presentation](https://XXXX)

## Contributors

* [Gionata Ghiggi](https://people.epfl.ch/gionata.ghiggi)
* [Kim Candolfi]
* [Jacopo Grazioli](https://people.epfl.ch/jacopo.grazioli) 
* [Alexis Berne]

## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
