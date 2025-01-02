# scMAE

scMAE: a masked autoencoder for single-cell RNA-seq clustering

## Overview

In scMAE, we randomly shuffle each gene with a certain probability and input them into the encoder to obtain low-dimensional representations. This shuffling of the input data serves the purpose of denoising and enables the model to learn the correlations between genes, resulting in more meaningful low-dimensional representations. A masking predictor is then employed to predict whether the expression values have been shuffled. Subsequently, the concatenated lowdimensional representations and masking prediction results are passed through the decoder to reconstruct the original gene expression matrix. The masking prediction results guide the decoder in identifying which gene values have been disrupted, facilitating a more accurate reconstruction of the original gene expression matrix.

## Installation

Clone this repository. The scMAE has been implemented in Python3.8.16 and Pytorch 1.13.1.

```
git clone https://github.com/CSUBioGroup/scMAE.git
cd scMAE/
```

## Datasets

All datasets used in our paper can be found in https://zenodo.org/deposit/8175767.

h5 file contains gene expression X and true label Y.

## Usage

We provided some demos to demonstrate usage of scMAE.

```Python
# args["dataset"].h5 save in args["paths"]["data"] directory (h5 file contains gene expression X and true label Y)
# hyperparameter
args = {}
args["dataset"] = "10X_PBMC" # dataset name
args['n_classes'] = 4 # number of clusters
args["paths"] = {"data": "/data/sc_data/all_data/", "results": "./res/"} # Datasets directory and output directory
args['batch_size'] = 256 
args["data_dim"] = 1000 # num_features of high variable genes
args['epochs'] = 80
args["num_workers"] = 4
args["learning_rate"] = 1e-3 
args["latent_dim"] = 32 # latent embedding dim
train(args)
```
