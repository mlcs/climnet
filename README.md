# climnet
Repository for creating and working with climate networks.


## Installation

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