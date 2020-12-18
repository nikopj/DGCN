# DGCN: Deep Graph Convolutional Network for image denoising.

This repository contains a PyTorch implementation of the paper [Deep Graph
Convoluitonal Image Denoising](https://arxiv.org/abs/1907.08448). 

## File Descriptions:
- `analyze.py`: contains functions for obtaining training curves, test-set
  performance, etc. given an argument file (i.e. `$ python3 analyze.py
  path/to/args.json`)
- `args.json`: sample arguments file used to interface with model configurations
  and checkpoints.
- `data.py`: defines datasets and data-loading functions.
- `knn.py`: defines functions for peforming K-nearest neighbors with
  local-masks.
- `net.py`: defines GCDN network and submodules.
- `train.py`: used like `analyze.py`, initializes and fits a model to training
  data given an arguments file.
- `utils.py`: defines functions for data pre/post processing, indexing, etc. 
- `visual.py`: visualization tools written with matplotlib package, such as an
  interactive receptive field viewer.

## Further Improvements:
- Faster nearest neighbors computations, perhaps with packages such as
  [FAISS](https://github.com/facebookresearch/faiss).


