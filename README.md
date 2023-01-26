[![DOI](https://zenodo.org/badge/464947192.svg)](https://zenodo.org/badge/latestdoi/464947192)

# climnet
Repository for creating and working with climate networks.


## Clone the repo and install all required packages

### 1. Clone repository with submodules:
```
git clone --recurse-submodules git@github.com:mlcs/climnet.git
```

### 2. Installing packages

Due to dependencies we recommend using conda. We provided a list of packages in the
'condaEnv.yml' file. The following steps set up a new environment with all required packages:
1. Install packages:
> conda env create -f condaEnv.yml
2. Activate environment:
> conda activate climnetenv
3. Install packages which are only available on pip
> pip install graphriccicurvature
3. Make your local version of climnet a package by running
> pip install -e .

### 3. Tutorial
A tutorial for reading data, processing and creating a simple correlation based climate network can be found at this ![tutorial](bin/tutorials/create_net.ipynb). 
