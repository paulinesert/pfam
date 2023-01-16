# PFAM 

PFAM is a Python library for performing prediction of the function of protein domains using the PFAM Dataset. 
It attempts to propose a methode to solve the Kaggle associated with the paper [[1]](#1) : https://www.kaggle.com/datasets/googleai/pfam-seed-random-split


DISCLAIMER : This repository is still WIP and not fully tested. 

## Method 

The method used in this repository is a reimplementation of the ProtCNN architecture presented in [[1]](#1). 

## Architecture overview 

Figure of the architecture to be added. 


## Installation

Download the data from the Kaggle page and store it in a data folder at the root of this directory. 
Install the requirements using either ```pip``` or ```conda```. 

```bash
pip install -r requirements.txt 
```
or 

```bash
conda env create -f environment.yml 
```

## Usage

Create a new (if needed) YAML config file that you will store in the config folder. 

Then to train, use the following command : 

```bash 
python train.py --config_file_path PATH/TO/YOUR/CONFIG_FILE
```

## Results on baseline config 

TBD. 

## Conclusions on the project 

TBD. 


## References
<a id="1">[1]</a> 
Bileschi et. al (2022). 
Using deep learning to annotate the protein universe
Nature

