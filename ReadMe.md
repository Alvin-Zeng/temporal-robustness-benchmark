# Benchmarking the Robustness of Temporal Action Detection Models Against Temporal Corruptions

## Introduction

Our paper is accepted to CVPR 2024 and an arXiv version can be found at [this link](https://arxiv.org/abs/2403.20254).

## Changelog

- 02/05/2024: We have uploaded the code for adding corruptions and extracting features on the Thumos14 dataset, including the I3D and videomae_v2 extractors. We have also included three example models trained using TRC loss.

## Table of Contents

The code includes:

- the addition of noise to I3D and videomaev2 features on the Thumos dataset, used for extracting features from the Thumos14-C (test set) or training models using TRC loss. 
- Three example models using TRC loss are included: actionformer, tridet, and temporalmaxer.

## Installation

- Read INSTALL.md for installing necessary dependencies.

## Quick Start

### Modify the configuration file

Before running the code, you need to modify the configuration file. For both extracting features and running the TRC loss example code, specific instructions for modifying the configuration file are provided at the beginning of each config file. Parameters that have `$$` in the comments need to be reset, while other parameters can be kept at their default values.

### Run the code

For I3D features on the Thumos dataset, run the command: `python main.py`

For videomaev2 features on the Thumos dataset, run the command: `python extract_tad_feature_thumos.py`