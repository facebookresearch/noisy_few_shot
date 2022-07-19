
# Few-shot Learning with Noisy Labels
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Authors: [Kevin J Liang](https://github.com/kevinjliang), Samrudhdhi B. Rangrej, Vladan Petrovic, Tal Hassner

This repository is the official PyTorch implementation of the [CVPR 2022](https://cvpr2022.thecvf.com/) paper [Few-shot Learning with Noisy Labels](https://arxiv.org/abs/2204.05494).

### Citation
If you find any part of our paper or this codebase useful, please consider citing our paper:

```
@inproceedings{liang2022few,
  title={Few-shot learning with noisy labels},
  author={Liang, Kevin J and Rangrej, Samrudhdhi B and Petrovic, Vladan and Hassner, Tal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9089--9098},
  year={2022}
}
```

### License
Please see [LICENSE.md](https://github.com/facebookresearch/noisy_few_shot/blob/main/LICENSE.md) for more details.

### Acknowledgements
This codebase was built starting from the [learn2learn](http://learn2learn.net/) library for meta-learning, in particular for setting up baseline non-noisy few-shot methods.

## Set-up

### Environment
For requirements, please see [`environment.yml`](https://github.com/facebookresearch/noisy_few_shot/blob/main/environment.yml). Experiments in the paper were run with Python 3.9.6 and Pytorch 1.9.0; other versions of Python and PyTorch may work, though are untested.

```
conda env create --file environment.yml
```

### Datasets
Place datasets in the directory `./data`. 
MiniImageNet and TieredImageNet datasets should download automatically when run for the first time if they do not exist already.
If running experiments with ImageNet outlier noise, you will need to generate the outlier dataset from the ImageNet training set. See [`create_miniimagenet_outliers.py`](https://github.com/facebookresearch/noisy_few_shot/blob/main/create_miniimagenet_outliers.py) for how to do so.


# Training and Evaluation
The script [`main.py`](https://github.com/facebookresearch/noisy_few_shot/blob/main/main.py) trains a model and then runs evaluation at the specified noise levels. Running just evaluation with this script can be done by simply setting the number of train epochs to 0 (`--max-epoch 0`).

This codebase uses [Visdom](https://github.com/fossasia/visdom) for visualizing training/eval curves. Although not required for the script to run, a Visdom server can be initialized with:

```
visdom
```

## Training backbone
For all models, we follow the common practice of first pre-training the backbone and then freezing it. An example command for training the conv4 backbone for MiniImageNet:

```
bash scripts/train_backbone.sh
```

This script will train the conv4 backbone and then evaluate it at the specified noise levels and noise type. This represents the performance of ProtoNet at various noise levels. Various other few-shot baselines can be run by loading this backbone and running evaluation on noisy support sets.

## Training/Evaluating TraNFS
In our paper, we propose a Transformer for Noisy Few-Shot (TraNFS), which learns a dynamic noise filtering mechanism for noisy support sets. An example command for launching TraNFS training for symmetric label swap noise on MiniImageNet:

```
bash scripts/train_transf.sh
```

After training the model, this script will also evaluate the model at the specified noise levels and noise type.


