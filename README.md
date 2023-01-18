# PFAM 

PFAM is a Python library for performing prediction of the function of protein domains using the PFAM Dataset. 
It attempts to propose a methode to solve the Kaggle associated with the paper [[1]](#1) : https://www.kaggle.com/datasets/googleai/pfam-seed-random-split

## Method 

The method used in this repository is a reimplementation of the ProtCNN architecture presented in [[1]](#1). 
1D-CNN are suited to deal with sequences (e.g. for machine translation, ..) and enable to take as inputs sequences of varying lengths. 
The use of dilated convolution layers in [[1]](#1) enables to model long-range dependencies as well as short ones. 

## Architecture overview 

The following figures illustrate the architecture and are taken from [[1]](#1).

![The sequence is first encoded as a sequence of one-hot vectors of amino acids. Then passed to an initial convolution and multiple residual blocks with dilated convolutions. Finally, a max-pooling along the sequence's lengths along with a linear layer and softmax activation yields the predicted probabilites.](https://github.com/paulinesert/pfam/blob/main/data/imgs/architecture.png)

![Each residual block is composed of a batch norm followed by a ReLU activation and then a dilated convolution. The output of the dilated convultion is then normalized using batch norm, passed through a ReLU activation and then to a bottleneck convolution whose purposes is to reduce the number of channels. A skip connection is then performed.](https://github.com/paulinesert/pfam/blob/main/data/imgs/residual_block.png)

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

Given more time and resources, one lead to explore would have been to research the state of the art to see how models that use attention (e.g. the encoder of Transformer model) could have been applied here  (with no causal masking since the whole sequence is known). One of the main challenge for those models would be to handle the sometimes very long protein sequence whose lengths are too big for current classic Transformer architecture. Splitting the sequence in sub-sequences of acceptable lengths for the current Transformer architecture could be one idea to tackle the problem but this would require discussions with biologists first to assess the validity of this scheme.

Another idea would have been to use a pre-trained model that models the protein sequences as language models do in NLP. 
Such models would be trained on a very large amount of protein sequences with a masking scheme such as in BERT (i.e. masking some amino acids and let the model predict them). ProteinBERT [[2]](#2) would be a starting point. 
Such model could later be used to perform the down-stream task of classifying the function of a protein sequence using the embeddings learnt by the pre-trained model.

## References
<a id="1">[1]</a> 
Bileschi et. al. (2022). 
Using deep learning to annotate the protein universe
Nature

<a id="2">[2]</a> 
Brandes et. al. (2022)
ProteinBERT: A universal deep-learning model of protein sequence and function
Bioinformatics, vol. 38